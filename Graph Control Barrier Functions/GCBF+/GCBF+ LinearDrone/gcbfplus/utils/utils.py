"""Small JAX array helper functions used by the simulation runtime."""


def merge01(x):
    """Merge the first two axes of an array, like einops ``(n1 n2)``."""
    if x.ndim < 2:
        raise ValueError(f"merge01 expects at least 2 dimensions, got shape {x.shape}")
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
