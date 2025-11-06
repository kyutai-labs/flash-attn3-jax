import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, xla
from jax.interpreters.mlir import ir

_flash_mha_kvcache_fwd_p = Primitive("flash_mha_kvcache_fwd")
_flash_mha_kvcache_fwd_p.multiple_results = True
_flash_mha_kvcache_fwd_p.def_impl(
    partial(xla.apply_primitive, _flash_mha_kvcache_fwd_p)
)


def _flash_mha_kvcache_fwd(
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    cache_seqlens,
    cache_batch_idx,
    cache_leftpad,
    page_table,
    rotary_cos,
    rotary_sin,
    softmax_scale,
    is_causal,
    window_size,
    num_splits,
):
    return tuple(
        _flash_mha_kvcache_fwd_p.bind(
            q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            cache_seqlens,
            cache_batch_idx,
            cache_leftpad,
            page_table,
            rotary_cos,
            rotary_sin,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size=window_size,
            num_splits=num_splits,
        )
    )


def ir_type_to_dtype(ty):
    for dtype in [np.dtype("bfloat16"), np.dtype("float16"), np.dtype("float32")]:
        if ty == mlir.dtype_to_ir_type(dtype):
            return dtype


def _flash_mha_kvcache_fwd_lowering(
    ctx,
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    cache_seqlens,
    cache_batch_idx,
    cache_leftpad,
    page_table,
    rotary_cos,
    rotary_sin,
    softmax_scale=None,
    is_causal=False,
    window_size=None,
    num_splits=1,
):
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    k_cache_type = ir.RankedTensorType(k_cache.type)
    k_cache_shape = k_cache_type.shape
    v_cache_type = ir.RankedTensorType(v_cache.type)
    v_cache_shape = v_cache_type.shape
    k_new_type = ir.RankedTensorType(k_new.type)
    k_new_shape = k_new_type.shape

    assert q_type.element_type == k_cache_type.element_type, (
        "Q and K must have the same dtype"
    )
    assert q_type.element_type == v_cache_type.element_type, (
        "Q and V must have the same dtype"
    )
    element_type = q_type.element_type
    assert type(element_type) in [ir.F16Type, ir.BF16Type], (
        "Only support fp16 and bf16 data type"
    )
    assert k_cache_shape == v_cache_shape, "K and V cache must have the same shape"
    [batch, seqlen_q, num_heads_q, head_dim] = q_shape

    # Check if we're using paged KV cache
    page_table_type = ir.RankedTensorType(page_table.type)
    page_table_shape = page_table_type.shape
    has_page_table = len(page_table_shape) == 2 and page_table_shape[0] > 0

    if has_page_table:
        [num_blocks, page_size, num_heads_k, head_dim_k] = k_cache_shape
        max_num_pages_per_seq = page_table_shape[1]
        seqlen_k_cache = max_num_pages_per_seq * page_size
    else:
        [batch_cache, seqlen_k_cache, num_heads_k, head_dim_k] = k_cache_shape

    assert head_dim == head_dim_k, "Q and K must have the same head dimension"
    assert isinstance(window_size, (tuple, list))

    # Check if we have new K/V
    has_k_new = len(k_new_shape) == 4 and k_new_shape[0] > 0

    # Check optional parameters
    cache_seqlens_type = ir.RankedTensorType(cache_seqlens.type)
    cache_batch_idx_type = ir.RankedTensorType(cache_batch_idx.type)
    cache_leftpad_type = ir.RankedTensorType(cache_leftpad.type)
    rotary_cos_type = ir.RankedTensorType(rotary_cos.type)

    has_cache_seqlens = cache_seqlens_type.shape[0] > 0
    has_cache_batch_idx = cache_batch_idx_type.shape[0] > 0
    has_cache_leftpad = cache_leftpad_type.shape[0] > 0
    has_rotary = rotary_cos_type.shape[0] > 0

    def fwd(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        cache_batch_idx,
        cache_leftpad,
        page_table,
        rotary_cos,
        rotary_sin,
    ):
        d = head_dim
        dpad = (8 - d % 8) % 8

        # Pad inputs if needed
        if dpad > 0:
            q = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, dpad)), "constant")
            k_cache = jnp.pad(k_cache, ((0, 0), (0, 0), (0, 0), (0, dpad)), "constant")
            v_cache = jnp.pad(v_cache, ((0, 0), (0, 0), (0, 0), (0, dpad)), "constant")
            if has_k_new:
                k_new = jnp.pad(k_new, ((0, 0), (0, 0), (0, 0), (0, dpad)), "constant")
                v_new = jnp.pad(v_new, ((0, 0), (0, 0), (0, 0), (0, dpad)), "constant")

        o_shape = [batch, seqlen_q, num_heads_q, d + dpad]
        lse_shape = [batch, num_heads_q, seqlen_q]

        # Allocate proper accumulator buffers when num_splits > 1
        if num_splits > 1:
            out_accum_shape = [num_splits, batch, num_heads_q, seqlen_q, d + dpad]
            lse_accum_shape = [num_splits, batch, num_heads_q, seqlen_q]
        else:
            # Dummy buffers for num_splits == 1
            out_accum_shape = [1, 1, 1, 1]
            lse_accum_shape = [1, 1, 1]

        jax_dtype = jnp.bfloat16 if type(element_type) == ir.BF16Type else jnp.float16
        out_types = [
            jax.ShapeDtypeStruct(o_shape, jax_dtype),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(out_accum_shape, jnp.float32),
            jax.ShapeDtypeStruct(lse_accum_shape, jnp.float32),
        ]

        empty_i32 = jnp.array([], dtype=jnp.int32)
        empty_f32 = jnp.array([], dtype=jnp.float32)
        empty_like_q = jnp.zeros((0,), dtype=jax_dtype)

        k_new_input = k_new if has_k_new else empty_like_q
        v_new_input = v_new if has_k_new else empty_like_q
        cache_seqlens_input = cache_seqlens if has_cache_seqlens else empty_i32
        cache_batch_idx_input = cache_batch_idx if has_cache_batch_idx else empty_i32
        cache_leftpad_input = cache_leftpad if has_cache_leftpad else empty_i32
        page_table_input = page_table if has_page_table else empty_i32
        rotary_cos_input = rotary_cos if has_rotary else empty_like_q
        rotary_sin_input = rotary_sin if has_rotary else empty_like_q

        # For paged KV cache, max_seqlen_k will be computed from page_table
        if has_page_table:
            max_seqlen_k_value = 0
            max_seqlen_k_has_value_flag = False
        else:
            max_seqlen_k_value = seqlen_k_cache
            max_seqlen_k_has_value_flag = True

        o, lse, _, _ = jax.ffi.ffi_call(
            "flash_mha_fwd",
            result_shape_dtypes=out_types,
            has_side_effect=False,
            input_layouts=[None] * 21,
            output_layouts=[None] * 4,
        )(
            q,
            k_cache,
            v_cache,
            k_new_input,  # k_new
            v_new_input,  # v_new
            empty_like_q,  # q_v
            empty_i32,  # cu_seqlens_q
            empty_i32,  # cu_seqlens_k
            empty_i32,  # cu_seqlens_k_new
            empty_i32,  # seqused_q
            cache_seqlens_input,  # seqused_k (cache_seqlens)
            page_table_input,  # page_table
            cache_batch_idx_input,  # kv_batch_idx
            cache_leftpad_input,  # leftpad_k
            rotary_cos_input,  # rotary_cos
            rotary_sin_input,  # rotary_sin
            empty_i32,  # seqlens_rotary
            empty_f32,  # q_descale
            empty_f32,  # k_descale
            empty_f32,  # v_descale
            empty_i32,  # scheduler_metadata
            max_seqlen_q=seqlen_q,
            max_seqlen_q_has_value=True,
            max_seqlen_k=max_seqlen_k_value,
            max_seqlen_k_has_value=max_seqlen_k_has_value_flag,
            softmax_scale_val=float(softmax_scale)
            if softmax_scale is not None
            else 1.0 / math.sqrt(d),
            softmax_scale_has_value=True,
            pack_gqa_val=False,
            pack_gqa_has_value=False,
            is_causal=is_causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=0,
            softcap=0.0,
            is_rotary_interleaved=True,
            num_splits=num_splits,
            sm_margin=0,
        )

        # Unpad head dimension if needed
        if dpad > 0:
            o = o[:, :, :, :d]

        return o, lse

    return mlir.lower_fun(fwd, multiple_results=True)(
        ctx,
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        cache_batch_idx,
        cache_leftpad,
        page_table,
        rotary_cos,
        rotary_sin,
    )


mlir.register_lowering(
    _flash_mha_kvcache_fwd_p,
    _flash_mha_kvcache_fwd_lowering,
    platform="gpu",
)


def _flash_mha_kvcache_fwd_abstract(
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    cache_seqlens,
    cache_batch_idx,
    cache_leftpad,
    page_table,
    rotary_cos,
    rotary_sin,
    softmax_scale=None,
    is_causal=None,
    window_size=None,
    num_splits=None,
):
    """Abstract evaluation for KV cache forward pass."""
    q_dtype = dtypes.canonicalize_dtype(q.dtype)
    k_cache_dtype = dtypes.canonicalize_dtype(k_cache.dtype)
    v_cache_dtype = dtypes.canonicalize_dtype(v_cache.dtype)
    [batch, seqlen_q, num_heads, head_dim] = q.shape
    assert q_dtype == k_cache_dtype and q_dtype == v_cache_dtype
    assert q_dtype in [jnp.bfloat16, jnp.float16]
    return (
        ShapedArray(q.shape, q_dtype),
        ShapedArray([batch, num_heads, seqlen_q], jnp.float32),
    )


_flash_mha_kvcache_fwd_p.def_abstract_eval(_flash_mha_kvcache_fwd_abstract)


def flash_mha_with_kvcache(
    q,
    k_cache,
    v_cache,
    k: jnp.ndarray | None = None,
    v: jnp.ndarray | None = None,
    cache_seqlens: int | jnp.ndarray | None = None,
    cache_batch_idx: jnp.ndarray | None = None,
    cache_leftpad: jnp.ndarray | None = None,
    page_table: jnp.ndarray | None = None,
    rotary_cos: jnp.ndarray | None = None,
    rotary_sin: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    is_causal: bool = False,
    window_size: tuple = (-1, -1),
    num_splits: int = 1,
    return_softmax_lse: bool = False,
):
    """Flash attention with KV cache supporting both contiguous and paged cache.

    Args:
        q: Query tensor of shape (batch, seqlen_q, num_heads, head_dim)
        k_cache: Key cache tensor. Shape depends on paging mode:
            - Contiguous: (batch_cache, seqlen_cache, num_heads_k, head_dim)
            - Paged: (num_blocks, page_size, num_heads_k, head_dim)
        v_cache: Value cache tensor. Shape depends on paging mode:
            - Contiguous: (batch_cache, seqlen_cache, num_heads_k, head_dim)
            - Paged: (num_blocks, page_size, num_heads_k, head_dim)
        k: Optional new keys to include in attention computation, shape (batch, seqlen_new, num_heads_k, head_dim)
            Attention is computed over concatenation of k_cache and k_new (cache is NOT mutated)
        v: Optional new values to include in attention computation, shape (batch, seqlen_new, num_heads_k, head_dim)
            Attention is computed over concatenation of v_cache and v_new (cache is NOT mutated)
        cache_seqlens: End position of valid data in the KV cache. Can be:
            - int: Same end position for all batch items
            - ndarray of shape (batch,): Per-batch end positions
            - None: Use full cache size
            Note: When used with cache_leftpad, the valid region is [cache_leftpad, cache_seqlens).
                  The length of valid data is (cache_seqlens - cache_leftpad).
            IMPORTANT: When providing new K/V (k and v parameters), the kernel requires
                  cache_seqlens + seqlen_new <= cache_size due to internal implementation.
                  This is a kernel limitation unrelated to cache mutation.
        cache_batch_idx: Optional indices to map query batch to cache batch,
            shape (batch,). If None, assumes identity mapping.
        cache_leftpad: Optional start position of valid data for each cache sequence, shape (batch,).
            Valid data is in the range [cache_leftpad, cache_seqlens).
        page_table: Optional page table for paged KV cache, shape (batch, max_num_pages_per_seq).
            Each entry contains the block index in the block pool for that sequence.
            If None, uses contiguous KV cache.
        rotary_cos: Optional rotary embedding cosine, shape (seqlen_ro, rotary_dim / 2)
        rotary_sin: Optional rotary embedding sine, shape (seqlen_ro, rotary_dim / 2)
        softmax_scale: Scaling factor for softmax. Defaults to 1/sqrt(head_dim)
        is_causal: Whether to apply causal masking
        window_size: Sliding window size (left, right). -1 means infinite window.
        num_splits: Number of splits for long sequences. Must be >= 1.
            Use num_splits=1 for single-pass attention (default for most cases),
            or higher values (4, 8, 16) for very long sequences.
        return_softmax_lse: Whether to return the logsumexp values

    Returns:
        out: Output tensor of shape (batch, seqlen_q, num_heads, head_dim)
        lse: Optional logsumexp tensor of shape (batch, num_heads, seqlen_q)
            Only returned if return_softmax_lse=True

    Notes:
        - This function does NOT support backward pass (forward-only inference)
        - Supports Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
        - Only supports float16 and bfloat16 dtypes
        - The KV cache is NOT modified (JAX arrays are immutable). When you provide k and v,
          attention is computed over the concatenation of cache and new K/V, but the cache
          tensor itself is never mutated.
        - When providing new K/V with cache_seqlens, ensure cache_seqlens + seqlen_new <= cache_size
          due to internal kernel requirements (the kernel needs space for temporary writes).
    """
    assert len(q.shape) == 4, "q must have shape (batch, seqlen_q, num_heads, head_dim)"
    assert len(k_cache.shape) == 4, (
        "k_cache must have shape (batch_cache, seqlen_cache, num_heads_k, head_dim) "
        "for contiguous mode or (num_blocks, page_size, num_heads_k, head_dim) for paged mode"
    )
    assert len(v_cache.shape) == 4, (
        "v_cache must have shape (batch_cache, seqlen_cache, num_heads_k, head_dim) "
        "for contiguous mode or (num_blocks, page_size, num_heads_k, head_dim) for paged mode"
    )
    assert k_cache.shape == v_cache.shape, (
        "k_cache and v_cache must have the same shape"
    )

    if page_table is not None:
        assert len(page_table.shape) == 2, (
            "page_table must have shape (batch, max_num_pages_per_seq)"
        )
        assert page_table.dtype == jnp.int32, "page_table must have dtype int32"

    assert q.dtype == k_cache.dtype == v_cache.dtype
    assert q.dtype in [jnp.bfloat16, jnp.float16], (
        "Only float16 and bfloat16 are supported"
    )

    if k is not None:
        assert v is not None, "If k is provided, v must also be provided"
        assert k.shape == v.shape, "k and v must have the same shape"
        assert k.dtype == q.dtype, "k must have the same dtype as q"
        assert len(k.shape) == 4, (
            "k must have shape (batch, seqlen_new, num_heads_k, head_dim)"
        )

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        batch_size = q.shape[0]
        cache_seqlens = jnp.full((batch_size,), cache_seqlens, dtype=jnp.int32)
    elif cache_seqlens is not None:
        cache_seqlens = jnp.asarray(cache_seqlens, dtype=jnp.int32)

    # Validate that new K/V fit within cache bounds when using cache_seqlens
    # Even though JAX arrays are immutable, the CUDA kernel internally writes new K/V
    # to the cache tensor starting at position cache_seqlens (for rotary embeddings, etc).
    # This requires cache_seqlens + seqlen_new <= cache_size to avoid out-of-bounds access.
    if k is not None and cache_seqlens is not None:
        seqlen_new = k.shape[1]
        max_cache_size = k_cache.shape[1]
        max_end_pos = (
            jnp.max(cache_seqlens).item()
            if hasattr(cache_seqlens, "shape")
            else cache_seqlens
        )

        if max_end_pos + seqlen_new > max_cache_size:
            raise ValueError(
                f"Invalid cache configuration: cache_seqlens ({max_end_pos}) + seqlen_new ({seqlen_new}) "
                f"= {max_end_pos + seqlen_new} exceeds cache size ({max_cache_size}). "
                f"The CUDA kernel requires cache_seqlens + seqlen_new <= cache_size to avoid "
                f"out-of-bounds memory access during computation. This is a kernel limitation, "
                f"not related to cache mutation (JAX arrays remain immutable)."
            )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    assert isinstance(softmax_scale, float)

    if num_splits < 1:
        raise ValueError(
            "num_splits must be >= 1. Use num_splits=1 for single-pass attention, "
            "or higher values (4, 8, 16) for splitting long sequences."
        )

    k_new = k if k is not None else jnp.zeros((0, 0, 0, 0), dtype=q.dtype)
    v_new = v if v is not None else jnp.zeros((0, 0, 0, 0), dtype=q.dtype)
    cache_seqlens_input = (
        cache_seqlens if cache_seqlens is not None else jnp.array([], dtype=jnp.int32)
    )
    cache_batch_idx_input = (
        cache_batch_idx
        if cache_batch_idx is not None
        else jnp.array([], dtype=jnp.int32)
    )
    cache_leftpad_input = (
        cache_leftpad if cache_leftpad is not None else jnp.array([], dtype=jnp.int32)
    )
    page_table_input = (
        page_table if page_table is not None else jnp.array([], dtype=jnp.int32)
    )
    rotary_cos_input = (
        rotary_cos if rotary_cos is not None else jnp.zeros((0,), dtype=q.dtype)
    )
    rotary_sin_input = (
        rotary_sin if rotary_sin is not None else jnp.zeros((0,), dtype=q.dtype)
    )

    out, lse = _flash_mha_kvcache_fwd(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens_input,
        cache_batch_idx_input,
        cache_leftpad_input,
        page_table_input,
        rotary_cos_input,
        rotary_sin_input,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        window_size=window_size,
        num_splits=num_splits,
    )

    return (out, lse) if return_softmax_lse else out
