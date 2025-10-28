"""Shared test utilities for flash attention tests."""

import jax
import jax.numpy as jnp
import pytest
from jax.tree_util import tree_map


def has_gpu():
    """Check if GPU is available."""
    try:
        return len(jax.devices("gpu")) > 0
    except:
        return False


def pretty(tensor):
    """Return a pretty string representation of tensor statistics."""
    shape = tensor.shape
    mx = jnp.max(tensor)
    mn = jnp.min(tensor)
    mean = jnp.mean(tensor)
    std = jnp.std(tensor)
    return f"[{shape}: {mn:.3g} | {mean:.3g}±{std:.3g} | {mx:.3g}]"


def check(ref_out, jax_out, out, margin=3):
    """Check that flash attention output matches reference within tolerance.

    Smart idea from Tri Dao's repo: compare both implementations to a float32
    reference implementation, and call it a pass if the absolute error isn't
    more than `margin` times worse with flash attention.

    Args:
        ref_out: Float32 reference output
        jax_out: Lower precision (float16/bfloat16) Jax reference output
        out: Flash attention output to check
        margin: Maximum allowed error ratio (default: 3)
    """

    def _check(ref_out, jax_out, out):
        assert (
            jnp.max(jnp.abs(out - ref_out)).item()
            <= margin * jnp.max(jnp.abs(jax_out - ref_out)).item()
        ), (pretty(jnp.abs(out - ref_out)), "vs", pretty(jnp.abs(jax_out - ref_out)))

    tree_map(_check, ref_out, jax_out, out)


def require_gpu():
    """Skip test if GPU is not available."""
    if not has_gpu():
        pytest.skip("GPU required for flash attention tests", allow_module_level=True)
