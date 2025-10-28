import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_default_matmul_precision", "highest")

from flash_attn3_jax import flash_mha

from .ref_mha import ref_mha
from .test_utils import check, has_gpu

if not has_gpu():
    pytest.skip("GPU required for flash attention tests", allow_module_level=True)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen_q", [97, 128])
@pytest.mark.parametrize("seqlen_k", [32, 63])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2])  # for MQA/GQA
def test_cross_fwd(n, seqlen_q, seqlen_k, h, d, m, dtype):
    q = jax.random.normal(
        jax.random.PRNGKey(0), [n, seqlen_q, h * m, d], dtype=jnp.float32
    )
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = ref_mha(q, k, v)
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref_mha(q, k, v)
    out = flash_mha(q, k, v)
    check(ref_out, jax_out, out, margin=3)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("d", [59, 32])
@pytest.mark.parametrize("h", [4])
@pytest.mark.parametrize("seqlen_q", [97, 128])
@pytest.mark.parametrize("seqlen_k", [32, 63])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("m", [1, 2])  # for MQA/GQA
def test_cross_bwd(n, seqlen_q, seqlen_k, h, d, m, dtype):
    @jax.grad
    def ref(qkv):
        return ref_mha(*qkv).sum()

    @jax.grad
    def flash(qkv):
        return flash_mha(*qkv).sum()

    q = jax.random.normal(
        jax.random.PRNGKey(0), [n, seqlen_q, h * m, d], dtype=jnp.float32
    )
    k = jax.random.normal(jax.random.PRNGKey(1), [n, seqlen_k, h, d], dtype=jnp.float32)
    v = jax.random.normal(jax.random.PRNGKey(2), [n, seqlen_k, h, d], dtype=jnp.float32)
    ref_out = ref((q, k, v))
    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)
    jax_out = ref((q, k, v))
    out = flash((q, k, v))
    check(ref_out, jax_out, out, margin=3)
