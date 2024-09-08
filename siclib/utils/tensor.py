"""
Author: Paul-Edouard Sarlin (skydes)
"""

import collections.abc as collections
import functools
import inspect
from typing import Callable, List, Tuple

import numpy as np
import torch

# flake8: noqa
# mypy: ignore-errors


string_classes = (str, bytes)


def autocast(func: Callable) -> Callable:
    """Cast the inputs of a TensorWrapper method to PyTorch tensors if they are numpy arrays.

    Use the device and dtype of the wrapper.

    Args:
        func (Callable): Method of a TensorWrapper class.

    Returns:
        Callable: Wrapped method.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    """Wrapper for PyTorch tensors."""

    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        """Wrapper for PyTorch tensors."""
        self._data = data

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying tensor."""
        return self._data.shape[:-1]

    @property
    def device(self) -> torch.device:
        """Get the device of the underlying tensor."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the underlying tensor."""
        return self._data.dtype

    def __getitem__(self, index) -> torch.Tensor:
        """Get the underlying tensor."""
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        """Set the underlying tensor."""
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        """Move the underlying tensor to a new device."""
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        """Move the underlying tensor to the CPU."""
        return self.__class__(self._data.cpu())

    def cuda(self):
        """Move the underlying tensor to the GPU."""
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        """Pin the underlying tensor to memory."""
        return self.__class__(self._data.pin_memory())

    def float(self):
        """Cast the underlying tensor to float."""
        return self.__class__(self._data.float())

    def double(self):
        """Cast the underlying tensor to double."""
        return self.__class__(self._data.double())

    def detach(self):
        """Detach the underlying tensor."""
        return self.__class__(self._data.detach())

    def numpy(self):
        """Convert the underlying tensor to a numpy array."""
        return self._data.detach().cpu().numpy()

    def new_tensor(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_tensor(*args, **kwargs)

    def new_zeros(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_zeros(*args, **kwargs)

    def new_ones(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_ones(*args, **kwargs)

    def new_full(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self._data.new_empty(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.unsqueeze(*args, **kwargs))

    def squeeze(self, *args, **kwargs):
        """Create a new tensor of the same type and device."""
        return self.__class__(self._data.squeeze(*args, **kwargs))

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        """Stack a list of objects with the same type and shape."""
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Support torch functions."""
        if kwargs is None:
            kwargs = {}
        return cls.stack(*args, **kwargs) if func is torch.stack else NotImplemented


def map_tensor(input_, func):
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif input_ is None:
        return None
    else:
        return func(input_)


def batch_to_numpy(batch):
    return map_tensor(batch, lambda tensor: tensor.cpu().numpy())


def batch_to_device(batch, device, non_blocking=True, detach=False):
    def _func(tensor):
        t = tensor.to(device=device, non_blocking=non_blocking, dtype=torch.float32)
        return t.detach() if detach else t

    return map_tensor(batch, _func)


def remove_batch_dim(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v for k, v in data.items()
    }


def add_batch_dim(data: dict) -> dict:
    """Add batch dimension to elements in data"""
    return {
        k: v[None] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def fit_to_multiple(x: torch.Tensor, multiple: int, mode: str = "center", crop: bool = False):
    """Get padding to make the image size a multiple of the given number.

    Args:
        x (torch.Tensor): Input tensor.
        multiple (int, optional): Multiple.
        crop (bool, optional): Whether to crop or pad. Defaults to False.

    Returns:
        torch.Tensor: Padding.
    """
    h, w = x.shape[-2:]

    if crop:
        pad_w = (w // multiple) * multiple - w
        pad_h = (h // multiple) * multiple - h
    else:
        pad_w = (multiple - w % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple

    if mode == "center":
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
    elif mode == "left":
        pad_l = 0
        pad_r = pad_w
        pad_t = 0
        pad_b = pad_h
    else:
        raise ValueError(f"Unknown mode {mode}")

    return (pad_l, pad_r, pad_t, pad_b)


def fit_features_to_multiple(
    features: torch.Tensor, multiple: int = 32, crop: bool = False
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad image to a multiple of the given number.

    Args:
        features (torch.Tensor): Input features.
        multiple (int, optional): Multiple. Defaults to 32.
        crop (bool, optional): Whether to crop or pad. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: Padded features and padding.
    """
    pad = fit_to_multiple(features, multiple, crop=crop)
    return torch.nn.functional.pad(features, pad, mode="reflect"), pad
