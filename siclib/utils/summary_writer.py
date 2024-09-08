"""This module implements the writer class for logging to tensorboard or wandb."""

import logging
import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig
from torch import nn
from torch.utils.tensorboard import SummaryWriter as TFSummaryWriter

from siclib import __module_name__

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    logger.debug("Could not import wandb.")
    wandb = None

# mypy: ignore-errors


def dot_conf(conf: DictConfig) -> Dict[str, Any]:
    """Recursively convert a DictConfig to a flat dict with keys joined by dots."""
    d = {}
    for k, v in conf.items():
        if isinstance(v, DictConfig):
            d |= {f"{k}.{k2}": v2 for k2, v2 in dot_conf(v).items()}
        else:
            d[k] = v
    return d


class SummaryWriter:
    """Writer class for logging to tensorboard or wandb."""

    def __init__(self, conf: DictConfig, args: DictConfig, log_dir: str):
        """Initialize the writer."""
        self.conf = conf

        if not conf.train.writer:
            self.use_wandb = False
            self.use_tensorboard = False
            return

        self.use_wandb = "wandb" in conf.train.writer
        self.use_tensorboard = "tensorboard" in conf.train.writer

        if self.use_wandb and not wandb:
            raise ImportError("wandb not installed.")

        if self.use_tensorboard:
            self.writer = TFSummaryWriter(log_dir=log_dir)

        if self.use_wandb:
            os.environ["WANDB__SERVICE_WAIT"] = "300"
            wandb.init(project=__module_name__, name=args.experiment, config=dot_conf(conf))

        if conf.train.writer and not self.use_wandb and not self.use_tensorboard:
            raise NotImplementedError(f"Writer {conf.train.writer} not implemented")

    def add_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value to tensorboard or wandb."""
        if self.use_wandb:
            step = 1 if step == 0 else step
            wandb.log({tag: value}, step=step)

        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

    def add_figure(self, tag: str, figure, step: Optional[int] = None):
        """Log a figure to tensorboard or wandb."""
        if self.use_wandb:
            step = 1 if step == 0 else step
            wandb.log({tag: figure}, step=step)
        if self.use_tensorboard:
            self.writer.add_figure(tag, figure, step)

    def add_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log a histogram to tensorboard or wandb."""
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)

    def add_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text to tensorboard or wandb."""
        if self.use_tensorboard:
            self.writer.add_text(tag, text, step)

    def add_pr_curve(self, tag: str, values, step: Optional[int] = None):
        """Log a precision-recall curve to tensorboard or wandb."""
        if self.use_wandb:
            step = 1 if step == 0 else step
            # @TODO: check if this works
            # wandb.log({"pr": wandb.plots.precision_recall(y_test, y_probas, labels)})
            wandb.log({tag: wandb.plots.precision_recall(values)}, step=step)

        if self.use_tensorboard:
            self.writer.add_pr_curve(tag, values, step)

    def watch(self, model: nn.Module, log_freq: int = 1000):
        """Watch a model for gradient updates."""
        if self.use_wandb:
            wandb.watch(
                model,
                log="gradients",
                log_freq=log_freq,
            )

    def close(self):
        """Close the writer."""
        if self.use_wandb:
            wandb.finish()

        if self.use_tensorboard:
            self.writer.close()
