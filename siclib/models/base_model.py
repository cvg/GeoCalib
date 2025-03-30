"""Base class for trainable models."""

import logging
import re
from abc import ABCMeta, abstractmethod
from copy import copy

import omegaconf
import torch
from omegaconf import OmegaConf
from torch import nn

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    logger.debug("Could not import wandb.")
    wandb = None

# flake8: noqa
# mypy: ignore-errors


class MetaModel(ABCMeta):
    def __prepare__(name, bases, **kwds):
        total_conf = OmegaConf.create()
        for base in bases:
            for key in ("base_default_conf", "default_conf"):
                update = getattr(base, key, {})
                if isinstance(update, dict):
                    update = OmegaConf.create(update)
                total_conf = OmegaConf.merge(total_conf, update)
        return dict(base_default_conf=total_conf)


class BaseModel(nn.Module, metaclass=MetaModel):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It recursively updates the default_conf of all parent classes, and
        it is updated by the user-provided configuration passed to __init__.
        Configurations can be nested.

        required_data_keys: list of expected keys in the input data dictionary.

        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.

        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.

        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """

    default_conf = {
        "name": None,
        "trainable": True,  # if false: do not optimize this model parameters
        "freeze_batch_normalization": False,  # use test-time statistics
        "timeit": False,  # time forward pass
        "watch": False,  # log weights and gradients to wandb
    }
    required_data_keys = []
    strict_conf = False

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(self.base_default_conf, OmegaConf.create(self.default_conf))
        if self.strict_conf:
            OmegaConf.set_struct(default_conf, True)

        # fixme: backward compatibility
        if "pad" in conf and "pad" not in default_conf:  # backward compat.
            with omegaconf.read_write(conf):
                with omegaconf.open_dict(conf):
                    conf["interpolation"] = {"pad": conf.pop("pad")}

        if isinstance(conf, dict):
            conf = OmegaConf.create(conf)
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf)

        # load pretrained weights
        if "weights" in conf and conf.weights is not None:
            logger.info(f"Loading checkpoint {conf.weights}")
            ckpt = torch.load(str(conf.weights), map_location="cpu", weights_only=False)
            weights_key = "model" if "model" in ckpt else "state_dict"
            self.flexible_load(ckpt[weights_key])

        if not conf.trainable:
            for p in self.parameters():
                p.requires_grad = False

        if conf.watch:
            try:
                wandb.watch(self, log="all", log_graph=True, log_freq=10)
                logger.info(f"Watching {self.__class__.__name__}.")
            except ValueError:
                logger.warning(f"Could not watch {self.__class__.__name__}.")

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Creating model {self.__class__.__name__} ({n_trainable/1e6:.2f} Mio)")

    def flexible_load(self, state_dict):
        """TODO: fix a probable nasty bug, and move to BaseModel."""
        # replace *gravity* with *up*
        for key in list(state_dict.keys()):
            if "gravity" in key:
                new_key = key.replace("gravity", "up")
                state_dict[new_key] = state_dict.pop(key)
                # print(f"Renaming {key} to {new_key}")

        # replace *_head.* with *_head.decoder.* for original paramnet checkpoints
        for key in list(state_dict.keys()):
            if "linear_pred_latitude" in key or "linear_pred_up" in key:
                continue

            if "_head" in key and "_head.decoder" not in key:
                # check if _head.{num} in key
                pattern = r"_head\.\d+"
                if re.search(pattern, key):
                    continue
                new_key = key.replace("_head.", "_head.decoder.")
                state_dict[new_key] = state_dict.pop(key)
                # print(f"Renaming {key} to {new_key}")

        dict_params = set(state_dict.keys())
        model_params = set(self.state_dict().keys())

        if dict_params == model_params:  # perfect fit
            logger.info("Loading all parameters of the checkpoint.")
            self.load_state_dict(state_dict, strict=True)
            return
        elif len(dict_params & model_params) == 0:  # perfect mismatch
            strip_prefix = lambda x: ".".join(x.split(".")[:1] + x.split(".")[2:])
            state_dict = {strip_prefix(n): p for n, p in state_dict.items()}
            dict_params = set(state_dict.keys())
            if len(dict_params & model_params) == 0:
                raise ValueError(
                    "Could not manage to load the checkpoint with"
                    "parameters:" + "\n\t".join(sorted(dict_params))
                )
        common_params = dict_params & model_params
        left_params = dict_params - model_params
        left_params = [
            p for p in left_params if "running" not in p and "num_batches_tracked" not in p
        ]
        logger.debug("Loading parameters:\n\t" + "\n\t".join(sorted(common_params)))
        if left_params:
            # ignore running stats of batchnorm
            logger.warning("Could not load parameters:\n\t" + "\n\t".join(sorted(left_params)))
        self.load_state_dict(state_dict, strict=False)

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        if self.conf.freeze_batch_normalization:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""

        def recursive_key_check(expected, given):
            for key in expected:
                assert key in given, f"Missing key {key} in data: {list(given.keys())}"
                if isinstance(expected, dict):
                    recursive_key_check(expected[key], given[key])

        recursive_key_check(self.required_data_keys, data)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        """To be implemented by the child class."""
        raise NotImplementedError
