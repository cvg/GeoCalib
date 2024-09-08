"""
Various handy Python and PyTorch utils.

Author: Paul-Edouard Sarlin (skydes)
"""

import os
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

# flake8: noqa
# mypy: ignore-errors


class AverageMetric:
    def __init__(self, elements=None):
        if elements is None:
            elements = []
            self._sum = 0
            self._num_examples = 0
        else:
            mask = ~np.isnan(elements)
            self._sum = sum(elements[mask])
            self._num_examples = len(elements[mask])

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        return np.nan if self._num_examples == 0 else self._sum / self._num_examples


# same as AverageMetric, but tracks all elements
class FAverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0
        self._elements = []

    def update(self, tensor):
        self._elements += tensor.cpu().numpy().tolist()
        assert tensor.dim() == 1, tensor.shape
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        return np.nan if self._num_examples == 0 else self._sum / self._num_examples


class MedianMetric:
    def __init__(self, elements=None):
        if elements is None:
            elements = []

        self._elements = elements

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan

        # set nan to inf to avoid error
        self._elements = np.array(self._elements)
        self._elements[np.isnan(self._elements)] = np.inf
        return np.nanmedian(self._elements)


class PRMetric:
    def __init__(self):
        self.labels = []
        self.predictions = []

    @torch.no_grad()
    def update(self, labels, predictions, mask=None):
        assert labels.shape == predictions.shape
        self.labels += (labels[mask] if mask is not None else labels).cpu().numpy().tolist()
        self.predictions += (
            (predictions[mask] if mask is not None else predictions).cpu().numpy().tolist()
        )

    @torch.no_grad()
    def compute(self):
        return np.array(self.labels), np.array(self.predictions)

    def reset(self):
        self.labels = []
        self.predictions = []


class QuantileMetric:
    def __init__(self, q=0.05):
        self._elements = []
        self.q = q

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanquantile(self._elements, self.q)


class RecallMetric:
    def __init__(self, ths, elements=None):
        if elements is None:
            elements = []

        self._elements = elements
        self.ths = ths

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        # set nan to inf to avoid error
        self._elements = np.array(self._elements)
        self._elements[np.isnan(self._elements)] = np.inf

        if isinstance(self.ths, Iterable):
            return [self.compute_(th) for th in self.ths]
        else:
            return self.compute_(self.ths[0])

    def compute_(self, th):
        if len(self._elements) == 0:
            return np.nan

        s = (np.array(self._elements) < th).sum()
        return s / len(self._elements)


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds, min_error: Optional[float] = None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e) / t
        aucs.append(np.round(auc, 4))
    return aucs


class AUCMetric:
    def __init__(self, thresholds, elements=None, min_error: Optional[float] = None):
        self._elements = elements
        self.thresholds = thresholds
        self.min_error = min_error
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1, tensor.shape
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan

        # set nan to inf to avoid error
        self._elements = np.array(self._elements)
        self._elements[np.isnan(self._elements)] = np.inf
        return compute_auc(self._elements, self.thresholds, self.min_error)


class Timer(object):
    """A simpler timer context object.
    Usage:
    ```
    > with Timer('mytimer'):
    >   # some computations
    [mytimer] Elapsed: X
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print(f"[{self.name}] Elapsed: {self.duration}")


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    import inspect

    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def set_num_threads(nt):
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl  # type: ignore
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    pth_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def fork_rng(seed=None, with_cuda=True):
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)


def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device
