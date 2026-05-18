import torch

from mobius import apply_mobius, inverse_mobius


def _pack_params(a, b, c, d):
    return torch.stack([a, b, c, d])


def apply_mobius_transform_torch(frame, a, b, c, d):
    theta = _pack_params(a, b, c, d)
    return apply_mobius(frame, theta, dim_names=["a", "b", "c", "d"])


def apply_inverse_mobius_transform_torch(frame, a, b, c, d):
    theta = _pack_params(a, b, c, d)
    return inverse_mobius(frame, theta, dim_names=["a", "b", "c", "d"])
