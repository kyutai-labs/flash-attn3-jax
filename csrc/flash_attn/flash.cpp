/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash.h"

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <stddef.h>

#include <cute/layout.hpp>

#include "check.h"
#include "cuda_check.h"
#include "fill_kernel.h"
#include "heuristics.h"
#include "static_switch.h"
#include "tile_size.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define PREPARE_VARLEN_MAX_BATCHES_1CTA 992

ffi::Error set_params_fprop(Flash_fwd_params &params, int32_t device, ffi::DataType element_type,
                            const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                            const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
                            const size_t h, const size_t h_k, const size_t d,
                            const size_t d_rounded, void *q_ptr, void *k_ptr, void *v_ptr,
                            void *out_ptr, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
                            void *seqused_q, void *seqused_k, void *softmax_lse_d, float p_dropout,
                            float softmax_scale, int window_size_left, int window_size_right,
                            int attention_chunk, const float softcap = 0.f, const int sm_margin = 0,
                            const size_t total_q = 0, const size_t total_k = 0) {
  // Reset the parameters
  params = {};

  params.is_bf16 = element_type == ffi::DataType::BF16;
  params.is_e4m3 = element_type == ffi::DataType::F8E4M3FN;

  // Set the pointers.
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = out_ptr;

  // Calculate strides assuming dense, row-major layout from JAX.
  // For varlen mode (cu_seqlens != nullptr), tensors have shape [total_seqlen, h, d]
  // For non-varlen mode, tensors have shape [b, seqlen, h, d]
  if (cu_seqlens_q_d != nullptr || cu_seqlens_k_d != nullptr) {
    // Varlen mode: tensors are [total_seqlen, h, d]
    // Use total_q/total_k for actual tensor dimensions
    auto q_s = cute::compact_row_major(cute::make_shape(total_q, h, d));
    auto k_s = cute::compact_row_major(cute::make_shape(total_k, h_k, d));
    auto v_s = cute::compact_row_major(cute::make_shape(total_k, h_k, d));
    auto o_s = cute::compact_row_major(cute::make_shape(total_q, h, d));

    params.q_row_stride = cute::get<0>(q_s);
    params.k_row_stride = cute::get<0>(k_s);
    params.v_row_stride = cute::get<0>(v_s);
    params.o_row_stride = cute::get<0>(o_s);

    params.q_head_stride = cute::get<1>(q_s);
    params.k_head_stride = cute::get<1>(k_s);
    params.v_head_stride = cute::get<1>(v_s);
    params.o_head_stride = cute::get<1>(o_s);
    params.v_dim_stride = cute::get<2>(v_s);
  } else {
    // Non-varlen mode: tensors are [b, seqlen, h, d]
    auto q_s = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
    auto k_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
    auto v_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
    auto o_s = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));

    params.q_row_stride = cute::get<1>(q_s);
    params.k_row_stride = cute::get<1>(k_s);
    params.v_row_stride = cute::get<1>(v_s);
    params.o_row_stride = cute::get<1>(o_s);

    params.q_head_stride = cute::get<2>(q_s);
    params.k_head_stride = cute::get<2>(k_s);
    params.v_head_stride = cute::get<2>(v_s);
    params.o_head_stride = cute::get<2>(o_s);
    params.v_dim_stride = cute::get<3>(v_s);

    params.q_batch_stride = cute::get<0>(q_s);
    params.o_batch_stride = cute::get<0>(o_s);
  }

  if (cu_seqlens_k_d == nullptr) {
    // Only set k/v batch strides if not using varlen k
    if (cu_seqlens_q_d == nullptr) {
      auto k_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
      auto v_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
      params.k_batch_stride = cute::get<0>(k_s);
      params.v_batch_stride = cute::get<0>(v_s);
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_q = static_cast<int *>(seqused_q);
  params.seqused_k = static_cast<int *>(seqused_k);
  params.softmax_lse_ptr = softmax_lse_d;

  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  params.scale_softmax = softmax_scale;
  params.softcap = softcap;
  params.p_dropout = 1.f - p_dropout;
  FFI_CHECK(p_dropout < 1.f) << "p_dropout must be < 1.0";
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  FFI_CHECK(p_dropout == 0.0f) << "This build does not support dropout.";
#endif
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;

  params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) &&
                    !params.is_causal;

  // TODO: check this
  if (window_size_left < 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  if (attention_chunk > 0) {
    window_size_left = std::min(window_size_left, attention_chunk - 1);
    window_size_right = std::min(window_size_right, attention_chunk - 1);
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.attention_chunk = attention_chunk;

  int major, minor, sm_count;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  params.arch = major * 10 + minor;
  params.num_sm = sm_count - sm_margin;
#ifdef FLASHATTENTION_DISABLE_LOCAL
  FFI_CHECK(!params.is_local) << "This build does not support local attention.";
#endif

  return ffi::Error::Success();
}

ffi::Error set_params_dgrad(Flash_bwd_params &params, int32_t device, ffi::DataType element_type,
                            const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                            const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
                            const size_t h, const size_t h_k, const size_t d,
                            const size_t d_rounded, void *q_ptr, void *k_ptr, void *v_ptr,
                            void *out_ptr, void *dout_ptr, void *dq_ptr, void *dk_ptr, void *dv_ptr,
                            void *cu_seqlens_q_d, void *cu_seqlens_k_d, void *seqused_q,
                            void *seqused_k, void *dq_accum_d, void *dk_accum_d, void *dv_accum_d,
                            void *softmax_lse_d, void *dsoftmax_sum_d, float p_dropout,
                            float softmax_scale, int window_size_left, int window_size_right,
                            int attention_chunk, const float softcap = 0.f,
                            bool deterministic = false, const int sm_margin = 0,
                            const size_t total_q = 0, const size_t total_k = 0) {
  set_params_fprop(params, device, element_type, b, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, h, h_k, d, d_rounded, q_ptr, k_ptr, v_ptr, out_ptr,
                   cu_seqlens_q_d, cu_seqlens_k_d, seqused_q, seqused_k, softmax_lse_d, p_dropout,
                   softmax_scale, window_size_left, window_size_right, attention_chunk, softcap,
                   sm_margin, total_q, total_k);

  params.do_ptr = dout_ptr;
  params.dq_ptr = dq_ptr;
  params.dk_ptr = dk_ptr;
  params.dv_ptr = dv_ptr;

  if (cu_seqlens_q_d != nullptr || cu_seqlens_k_d != nullptr) {
    // Varlen mode: tensors are [total_seqlen, h, d]
    auto dout_s = cute::compact_row_major(cute::make_shape(total_q, h, d));
    auto dq_s = cute::compact_row_major(cute::make_shape(total_q, h, d));
    auto dk_s = cute::compact_row_major(cute::make_shape(total_k, h_k, d));
    auto dv_s = cute::compact_row_major(cute::make_shape(total_k, h_k, d));

    params.do_row_stride = cute::get<0>(dout_s);
    params.dq_row_stride = cute::get<0>(dq_s);
    params.dk_row_stride = cute::get<0>(dk_s);
    params.dv_row_stride = cute::get<0>(dv_s);

    params.do_head_stride = cute::get<1>(dout_s);
    params.dq_head_stride = cute::get<1>(dq_s);
    params.dk_head_stride = cute::get<1>(dk_s);
    params.dv_head_stride = cute::get<1>(dv_s);
  } else {
    // Non-varlen mode: tensors are [b, seqlen, h, d]
    auto dout_s = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
    auto dq_s = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
    auto dk_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));
    auto dv_s = cute::compact_row_major(cute::make_shape(b, seqlen_k, h_k, d));

    params.do_row_stride = cute::get<1>(dout_s);
    params.dq_row_stride = cute::get<1>(dq_s);
    params.dk_row_stride = cute::get<1>(dk_s);
    params.dv_row_stride = cute::get<1>(dv_s);

    params.do_head_stride = cute::get<2>(dout_s);
    params.dq_head_stride = cute::get<2>(dq_s);
    params.dk_head_stride = cute::get<2>(dk_s);
    params.dv_head_stride = cute::get<2>(dv_s);

    params.do_batch_stride = cute::get<0>(dout_s);
    params.dq_batch_stride = cute::get<0>(dq_s);
    params.dk_batch_stride = cute::get<0>(dk_s);
    params.dv_batch_stride = cute::get<0>(dv_s);
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;
  params.dsoftmax_sum = dsoftmax_sum_d;
  params.deterministic = deterministic;

  return ffi::Error::Success();
}

template <int Arch, int Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params &params, cudaStream_t stream) {
  if (!params.is_e4m3) {
    if (params.is_bf16) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
      if (params.d <= 64) {
#ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
        if constexpr (Arch == 90) {
          if (params.dv > 256) {
            return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA,
                                Has_softcap, PackGQA>(params, stream);
          } else if (params.dv > 64) {
            return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA,
                                Has_softcap, PackGQA>(params, stream);
          }
        }
#endif
        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
      if (params.d <= 96) {
        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
      if (params.d <= 128) {
        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
      if (params.d <= 192) {
#ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
        if constexpr (Arch == 90) {
          if (params.dv <= 128) {
            return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA,
                                Has_softcap, PackGQA>(params, stream);
          }
        }
#endif
        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
      if (params.d <= 256) {
        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
    } else {
#ifndef FLASHATTENTION_DISABLE_FP16
#ifndef FLASHATTENTION_DISABLE_HDIM64
      if (params.d <= 64) {
#ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
        if constexpr (Arch == 90) {
          if (params.dv > 256) {
            return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap,
                                PackGQA>(params, stream);
          } else if (params.dv > 64) {
            return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap,
                                PackGQA>(params, stream);
          }
        }
#endif
        return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
      if (params.d <= 96) {
        return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
      if (params.d <= 128) {
        return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
      if (params.d <= 192) {
#ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
        if constexpr (Arch == 90) {
          if (params.dv <= 128) {
            return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap,
                                PackGQA>(params, stream);
          }
        }
#endif
        return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
      if (params.d <= 256) {
        return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap,
                            PackGQA>(params, stream);
      }
#endif
#else
      throw std::runtime_error("This flash attention build does not support FP16.");
#endif
    }
  } else {
#ifndef FLASHATTENTION_DISABLE_FP8
#ifndef FLASHATTENTION_DISABLE_HDIM64
    if (params.d <= 64) {
      return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap,
                          PackGQA>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
    if (params.d <= 96) {
      return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap,
                          PackGQA>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
    if (params.d <= 128) {
      return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap,
                          PackGQA>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
    if (params.d <= 192) {
#ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
      if constexpr (Arch == 90) {
        if (params.dv <= 128) {
          return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA,
                              Has_softcap, PackGQA>(params, stream);
        }
      }
#endif
      return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap,
                          PackGQA>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
    if (params.d <= 256) {
      return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap,
                          PackGQA>(params, stream);
    }
#endif
#else
    throw std::runtime_error("This flash attention build does not support FP8.");
#endif
  }
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  // HEADDIM_SWITCH(params.d, [&] {
  //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
  // });
  ARCH_SWITCH(params.arch, Arch, [&] {
    SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
      PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
        PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
          // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
          static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
          SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
            run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
          });
        });
      });
    });
  });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl = false) {
#ifndef FLASHATTENTION_DISABLE_SPLIT
  // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
  // so that kBlockM is smaller and we have more parallelism.
  if (params.is_fp32) {
    if (params.dv <= 64) {
      run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
    } else {
      run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
    }
  } else if (params.is_bf16) {
    if (params.dv <= 64) {
      run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
    } else {
      run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
    }
  } else {
    if (params.dv <= 64) {
      run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
    } else {
      run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
    }
  }
#else
  FFI_CHECK(false) << "This flash attention build does not support combine kernels.";
#endif
}

inline bool get_pagedkv_tma(Flash_fwd_params const &params) {
  if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) {
    return false;
  }
  // This needs to match the kernel configs
  auto kBlockMN_kernel_args_sm90 =
      tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
                         params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/,
                         false /*paged_kv_non_TMA*/, params.softcap > 0.f);
  int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
  int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
  // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
  // at least for MLA.
  return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const &params) {
  // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
  // Has little effect on speed.
  if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) {
    return true;
  }
#ifdef FLASHATTENTION_DISABLE_PACKGQA
  return false;
#else
  // params.page_table must already be set
  if (params.h == params.h_k) {
    return false;
  }
  // This needs to match the kernel configs
  auto kBlockMN_kernel_args_sm90 =
      tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
                         params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/,
                         params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
  int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
  return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q,
                         params.h / params.h_k, kBlockM);
#endif
}

inline int get_num_splits(Flash_fwd_params const &params) {
#ifdef FLASHATTENTION_DISABLE_SPLIT
  return 1;
#else
  // Always enable PackGQA for Split
  // params.page_table must already be set
  // This needs to match the kernel configs
  bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q ||
                params.seqused_k || params.leftpad_k;
  auto kBlockMN_kernel_args_sm90 =
      tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
                         params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/,
                         params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
  // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
  // has not been set here. It's OK though because we might just underestimate kBlockN a bit
  auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(
      params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal,
      params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen,
      params.softcap > 0.f, params.knew_ptr);
  int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90)
                                        : std::get<0>(kBlockMN_kernel_args_sm8x);
  int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90)
                                        : std::get<1>(kBlockMN_kernel_args_sm8x);
  int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
  // If is_local, we're not going to load all of seqlen_k
  int const seqlen_k_loaded =
      !params.is_local
          ? params.seqlen_k
          : std::max(0, std::min(params.seqlen_k,
                                 params.window_size_right + params.window_size_left + 1 + kBlockM));
  int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
  int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
  int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
  // Always enable PackGQA for Split
  // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on
  // num_splits. We assume the case where there's 1 long sequence and the rest are short, i.e.
  // pretending that batch = 1.
  int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
  return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks,
                              size_one_kv_head, params.is_causal || params.is_local, 128);
#endif
}

inline int get_max_headdim() {
#ifndef FLASHATTENTION_DISABLE_HDIM256
  return 256;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  return 192;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  return 128;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  return 96;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM64
  return 64;
#endif
  return 0;
}

inline int round_up_headdim(int head_size) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) {
    return 64;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  if (head_size <= 96) {
    return 96;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) {
    return 128;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  if (head_size <= 192) {
    return 192;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
  if (head_size <= 256) {
    return 256;
  }
#endif
  return 256;
}

inline int round_up_headdimv(int head_size) {
  if (head_size <= 64) {
    return 64;
  }
  if (head_size <= 96) {
    return 96;
  }
  if (head_size <= 128) {
    return 128;
  }
  if (head_size <= 192) {
    return 192;
  }
  if (head_size <= 256) {
    return 256;
  }
  return 512;
}

ffi::Error mha_fwd_get_scheduler_metadata_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, int64_t batch_size,
    int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads, int64_t num_heads_k,
    int64_t headdim, int64_t headdim_v, int32_t qkv_dtype_int, ffi::AnyBuffer seqused_k,
    ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer cu_seqlens_k_new_,
    ffi::AnyBuffer seqused_q_, ffi::AnyBuffer leftpad_k_, int64_t page_size_val,
    bool page_size_has_value, int64_t max_seqlen_k_new, bool is_causal, int64_t window_size_left,
    int64_t window_size_right, int64_t attention_chunk, bool has_softcap, int64_t num_splits,
    bool pack_gqa_val, bool pack_gqa_has_value, int64_t sm_margin,
    ffi::Result<ffi::AnyBuffer> tile_count_semaphore) {
  auto qkv_dtype = static_cast<ffi::DataType>(qkv_dtype_int);
  FFI_CHECK(qkv_dtype == ffi::DataType::F16 || qkv_dtype == ffi::DataType::BF16 ||
            qkv_dtype == ffi::DataType::F8E4M3FN)
      << "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type";
  FFI_CHECK(num_heads % num_heads_k == 0)
      << "Number of heads in key/value must divide number of heads in query";

  Flash_fwd_params params{};
  params.is_bf16 = qkv_dtype == ffi::DataType::BF16;
  params.is_e4m3 = qkv_dtype == ffi::DataType::F8E4M3FN;
  params.b = batch_size;
  params.seqlen_q = max_seqlen_q;
  params.seqlen_k = max_seqlen_k;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.d = headdim;
  params.dv = headdim_v;
  params.d_rounded = round_up_headdim(headdim);
  params.dv_rounded = headdim_v == headdim ? params.d_rounded : round_up_headdimv(headdim_v);
  params.seqlen_knew = max_seqlen_k_new;

  params.cu_seqlens_q =
      (cu_seqlens_q_.element_count() > 0) ? cu_seqlens_q_.typed_data<int>() : nullptr;
  params.cu_seqlens_k =
      (cu_seqlens_k_.element_count() > 0) ? cu_seqlens_k_.typed_data<int>() : nullptr;
  params.cu_seqlens_knew =
      (cu_seqlens_k_new_.element_count() > 0) ? cu_seqlens_k_new_.typed_data<int>() : nullptr;
  params.seqused_q = (seqused_q_.element_count() > 0) ? seqused_q_.typed_data<int>() : nullptr;
  params.seqused_k = seqused_k.typed_data<int>();
  params.leftpad_k = (leftpad_k_.element_count() > 0) ? leftpad_k_.typed_data<int>() : nullptr;
  params.knew_ptr = params.seqlen_knew > 0 ? reinterpret_cast<int *>(1) : nullptr;
  if (window_size_left >= max_seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= max_seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (max_seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 &&
      attention_chunk == 0) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
    if ((headdim <= 64 || headdim > 128) || !page_size_has_value) {
      is_causal = false;
    }
  }
  if (is_causal) {
    window_size_right = 0;
  }

  params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) &&
                    !params.is_causal;
  if (window_size_left < 0) {
    window_size_left = max_seqlen_k - 1;
  }
  if (window_size_right < 0) {
    window_size_right = max_seqlen_q - 1;
  }
  if (attention_chunk > 0) {
    window_size_left = std::min(window_size_left, attention_chunk - 1);
    window_size_right = std::min(window_size_right, attention_chunk - 1);
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.attention_chunk = attention_chunk;
  // Set device properties from CUDA runtime
  int major, minor, sm_count;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  params.arch = major * 10 + minor;
  params.num_sm = sm_count - sm_margin;

  params.softcap = has_softcap ? 1.0f : 0.0f;
  params.page_size = (page_size_val > 0) ? page_size_val : 1;
  params.page_table = (page_size_val > 0) ? reinterpret_cast<int *>(1) : nullptr;

  bool const use_prepare_varlen = true;
  params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
  params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int *>(1);

  params.pagedkv_tma = get_pagedkv_tma(params);
  params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
  // Always enable PackGQA for Split, and get_pack_gqa requires params.num_splits to decide
  params.pack_gqa = pack_gqa_has_value ? pack_gqa_val : get_pack_gqa(params);
  bool is_varlen = true;

  bool const scheduler_needs_semaphore =
      params.arch >= 90
          ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
          : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
  params.varlen_sort_batches = !params.is_local; // Use this value for Sort in scheduler template
  params.head_swizzle =
      params.is_causal || params.is_local; // Use this value for LPT in scheduler template
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int tile_count_semaphore_offset = 0;
  if (scheduler_needs_semaphore || use_prepare_varlen) {
    int b_rounded = round_multiple(params.b, 4);
    int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
    if (params.varlen_sort_batches) {
      num_prepare_batch_vectors += 1;
    }
    if (params.head_swizzle) {
      num_prepare_batch_vectors += 1;
    }
    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    int *tile_count_semaphore_base_ptr = tile_count_semaphore->typed_data<int>();
    params.num_splits_dynamic_ptr = use_prepare_varlen ? tile_count_semaphore_base_ptr : nullptr;
    params.num_m_blocks_ptr =
        use_prepare_varlen ? tile_count_semaphore_base_ptr + b_rounded : nullptr;
    params.varlen_batch_idx_ptr = use_prepare_varlen && params.varlen_sort_batches
                                      ? tile_count_semaphore_base_ptr + b_rounded * 2
                                      : nullptr;
    // params.num_n_blocks_ptr  = use_prepare_varlen && params.head_swizzle ?
    // tile_count_semaphore.data_ptr<int>() + head_swizzle_offset : nullptr;
    params.num_nheads_in_l2_ptr = use_prepare_varlen && params.head_swizzle
                                      ? tile_count_semaphore_base_ptr + head_swizzle_offset
                                      : nullptr;
    if (scheduler_needs_semaphore) {
      if (!use_prepare_varlen) {
        FFI_CUDA_CHECK(cudaMemsetAsync(tile_count_semaphore->untyped_data(), 0,
                                       tile_count_semaphore->size_bytes(), stream));

      } // If varlen we'll manually do the zero-ing
      params.tile_count_semaphore = tile_count_semaphore_base_ptr + tile_count_semaphore_offset;

    } else {
      params.tile_count_semaphore = nullptr;
    }
  }
  if (use_prepare_varlen) {
    auto kBlockMN_kernel_args_sm90 =
        tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
                           params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/,
                           params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(
        params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded,
        params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/,
        params.page_table, is_varlen && params.num_splits > 1, params.softcap > 0.f,
        params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90)
                                          : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90)
                                          : std::get<1>(kBlockMN_kernel_args_sm8x);
    prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN,
                              false /*enable_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();
  }
  return ffi::Error();
}

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
ffi::Error
mha_fwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device, ffi::AnyBuffer q,
             ffi::AnyBuffer k, ffi::AnyBuffer v, ffi::AnyBuffer k_new_, ffi::AnyBuffer v_new_,
             ffi::AnyBuffer q_v_, ffi::AnyBuffer cu_seqlens_q_, ffi::AnyBuffer cu_seqlens_k_,
             ffi::AnyBuffer cu_seqlens_k_new_, ffi::AnyBuffer seqused_q_, ffi::AnyBuffer seqused_k_,
             ffi::AnyBuffer page_table_, ffi::AnyBuffer kv_batch_idx_, ffi::AnyBuffer leftpad_k_,
             ffi::AnyBuffer rotary_cos_, ffi::AnyBuffer rotary_sin_, ffi::AnyBuffer seqlens_rotary_,
             ffi::AnyBuffer q_descale_, ffi::AnyBuffer k_descale_, ffi::AnyBuffer v_descale_,
             ffi::AnyBuffer scheduler_metadata_, int64_t max_seqlen_q_val,
             bool max_seqlen_q_has_value, int64_t max_seqlen_k_val, bool max_seqlen_k_has_value,
             double softmax_scale_val, bool softmax_scale_has_value, bool pack_gqa_val,
             bool pack_gqa_has_value, bool is_causal, int64_t window_size_left,
             int64_t window_size_right, int64_t attention_chunk, double softcap,
             bool is_rotary_interleaved, int64_t num_splits, int64_t sm_margin,
             ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> softmax_lse,
             ffi::Result<ffi::AnyBuffer> out_accum, ffi::Result<ffi::AnyBuffer> softmax_lse_accum) {
  int major;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CHECK(major >= 8) << "FlashAttention only supports Ampere GPUs or newer.";

  auto dtype = q.element_type();
  FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16 ||
            dtype == ffi::DataType::F8E4M3FN)
      << ffi::ErrorCode::kInvalidArgument
      << "FlashAttention only supports fp16, bf16, and fp8_e4m3fn data type";
  if (major < 9) {
    FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
        << ffi::ErrorCode::kInvalidArgument
        << "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type";
  }
  FFI_CHECK(k.element_type() == dtype && v.element_type() == dtype)
      << "query, key, and value must have the same dtype";

  // Check for optional arguments by checking buffer size as JAX FFI cannot pass null pointers
  // We use 0 sized buffers for optional arguments
  const bool is_varlen_q = cu_seqlens_q_.element_count() > 0;
  const bool is_varlen_k = cu_seqlens_k_.element_count() > 0;
  const bool paged_KV = page_table_.element_count() > 0;
  const bool has_k_new = k_new_.element_count() > 0;

  if (paged_KV) {
    FFI_CHECK(page_table_.element_type() == ffi::DataType::S32)
        << "page_table must have S32 (int32) dtype.";
  }
  if (is_varlen_q) {
    FFI_CHECK(cu_seqlens_q_.element_type() == ffi::DataType::S32);
    FFI_CHECK(max_seqlen_q_val > 0) << "max_seqlen_q must be provided if cu_seqlens_q is provided";
  }
  if (is_varlen_k) {
    FFI_CHECK(cu_seqlens_k_.element_type() == ffi::DataType::S32);
    FFI_CHECK(max_seqlen_k_val > 0) << "max_seqlen_k must be provided if cu_seqlens_k is provided";
    FFI_CHECK(!paged_KV) << "If cu_seqlens_k is passed, paged KV is not supported";
    FFI_CHECK(kv_batch_idx_.element_count() <= 0)
        << "If cu_seqlens_k is passed, kv_batch_idx is not supported";
  }

  // Shape calculations
  auto const sizes = q.dimensions();
  int const batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q_.dimensions()[0] - 1;
  int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_val;
  int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  // For varlen: q shape is [total_q, h, d], otherwise [b, s_q, h, d]
  int num_heads = !is_varlen_q ? sizes[2] : sizes[1];
  int const head_size = !is_varlen_q ? sizes[3] : sizes[2];
  int const head_size_v = !is_varlen_q ? v.dimensions()[3] : v.dimensions()[2];
  int const max_num_pages_per_seq = !paged_KV ? 0 : page_table_.dimensions()[1];
  int const num_pages = !paged_KV ? 0 : k.dimensions()[0];
  int const page_size = !paged_KV ? 1 : k.dimensions()[1];
  int const seqlen_k = !is_varlen_k
                           ? (!paged_KV ? k.dimensions()[1] : max_num_pages_per_seq * page_size)
                           : max_seqlen_k_val;
  int const total_k = !is_varlen_k ? batch_size * k.dimensions()[1] : k.dimensions()[0];
  // For varlen: k shape is [total_k, h_k, d], otherwise [b, s_k, h_k, d]
  int const num_heads_k = !is_varlen_k ? k.dimensions()[2] : k.dimensions()[1];
  int const batch_size_k =
      !paged_KV ? (!is_varlen_k ? k.dimensions()[0] : cu_seqlens_k_.dimensions()[0] - 1)
                : page_table_.dimensions()[0];
  double softmax_scale =
      softmax_scale_has_value ? softmax_scale_val : (1.0 / sqrt(static_cast<double>(head_size)));

  if (kv_batch_idx_.element_count() <= 0) {
    FFI_CHECK(batch_size == batch_size_k) << "batch_size must be equal to batch_size_k";
  }
  int const max_headdim = get_max_headdim();
  FFI_CHECK(head_size <= max_headdim)
      << "FlashAttention forward only supports head dimension at most " +
             std::to_string(max_headdim);
  FFI_CHECK(num_heads % num_heads_k == 0)
      << "Number of heads in key/value must divide number of heads in query";

  if (head_size_v != head_size) {
    FFI_CHECK((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) ||
              (head_size <= 64 && head_size_v <= 512))
        << "If V headdim is different from Q/K dim, we only support Q/K headdim in (128, "
           "192] "
           "and V headdim in (96, 128], "
           "or (Q/K <= 64 and V <= 512).";
    FFI_CHECK(major == 9) << "Only Hopper supports different V headdim";
    if (head_size_v > 256) {
      FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
          << "HeaddimV > 256 requires fp16 and bf16 data type";
    }
  }

  // This needs to go before kBlockM & kBlockN since we rely on the correct window_size and
  // is_causal to set kBlockM
  // TODO: check this
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1 && attention_chunk == 0) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
    if ((head_size <= 64 || head_size > 128) || !paged_KV) {
      is_causal = false;
    }
  }
  if (is_causal) {
    window_size_right = 0;
  }

  bool const seqused_q_has_value = seqused_q_.element_count() > 0;
  bool const seqused_k_has_value = seqused_k_.element_count() > 0;
  bool const leftpad_k_has_value = leftpad_k_.element_count() > 0;
  if (seqused_q_has_value) {
    FFI_CHECK(seqused_q_.element_type() == ffi::DataType::S32) << "seqused_q must have dtype int32";
  }
  if (seqused_k_has_value) {
    FFI_CHECK(seqused_k_.element_type() == ffi::DataType::S32) << "seqused_k must have dtype int32";
  }

  if (leftpad_k_has_value) {
    FFI_CHECK(leftpad_k_.element_type() == ffi::DataType::S32) << "leftpad_k must have dtype int32";
  }
  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_has_value || seqused_k_has_value ||
                         leftpad_k_has_value;
#ifdef FLASHATTENTION_DISABLE_VARLEN
  FFI_CHECK(!is_varlen) << "This flash attention build does not support varlen.";
#endif

  int const alignment = dtype == ffi::DataType::F8E4M3FN ? 16 : 8;
  FFI_CHECK(head_size % alignment == 0)
      << "head_size should be a multiple of " + std::to_string(alignment);
  FFI_CHECK(head_size_v % alignment == 0)
      << "head_size_v should be a multiple of " + std::to_string(alignment);

  // Set up params
  Flash_fwd_params params;
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded =
      head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Main parameter setup
  set_params_fprop(params, device, dtype, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
                   q.untyped_data(), k.untyped_data(), v.untyped_data(), out->untyped_data(),
                   !is_varlen_q ? nullptr : cu_seqlens_q_.typed_data<int>(),
                   !is_varlen_k ? nullptr : cu_seqlens_k_.typed_data<int>(),
                   seqused_q_has_value ? seqused_q_.typed_data<int>() : nullptr,
                   seqused_k_has_value ? seqused_k_.typed_data<int>() : nullptr,
                   softmax_lse->typed_data<float>(), 0.f, softmax_scale, window_size_left,
                   window_size_right, attention_chunk, softcap, sm_margin, total_q, total_k);

  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.dv_rounded = head_size_v_rounded;
  params.is_bf16 = (dtype == ffi::DataType::BF16);

  if (leftpad_k_has_value)
    params.leftpad_k = leftpad_k_.typed_data<int>();
  if (paged_KV) {
    params.page_table = page_table_.typed_data<int>();
    auto pts = cute::compact_row_major(
        cute::make_shape(page_table_.dimensions()[0], page_table_.dimensions()[1]));
    params.page_table_batch_stride = cute::get<0>(pts);

    // For paged KV cache, recalculate strides based on actual tensor shape
    // k/v have shape (num_blocks, page_size, h_k, d) not (b, seqlen_k, h_k, d)
    // Use negative indexing like PyTorch: stride(-3) = page_size stride, stride(-2) = h_k stride, etc.
    auto k_dims = k.dimensions();
    auto v_dims = v.dimensions();
    auto k_paged_s = cute::compact_row_major(
        cute::make_shape(k_dims[0], k_dims[1], k_dims[2], k_dims[3]));
    auto v_paged_s = cute::compact_row_major(
        cute::make_shape(v_dims[0], v_dims[1], v_dims[2], v_dims[3]));

    // Override the strides that were incorrectly calculated in set_params_fprop
    params.k_row_stride = cute::get<1>(k_paged_s);      // stride for page_size dim
    params.v_row_stride = cute::get<1>(v_paged_s);      // stride for page_size dim
    params.k_head_stride = cute::get<2>(k_paged_s);     // stride for h_k dim
    params.v_head_stride = cute::get<2>(v_paged_s);     // stride for h_k dim
    params.v_dim_stride = cute::get<3>(v_paged_s);      // stride for d dim
    params.k_batch_stride = cute::get<0>(k_paged_s);    // stride for num_blocks dim
    params.v_batch_stride = cute::get<0>(v_paged_s);    // stride for num_blocks dim
  }
  params.page_size = page_size;
  params.num_pages = num_pages;

  // Handle k_new and v_new for append-to-cache
  if (has_k_new) {
    FFI_CHECK(v_new_.element_count() > 0) << "If k_new is supplied, v_new must also be passed in";
    FFI_CHECK(seqused_k_.element_count() > 0)
        << "If k_new is supplied, seqlens_k must also be passed in";
    FFI_CHECK(seqlen_q <= seqlen_k)
        << "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache";

    const bool is_varlen_k_new = cu_seqlens_k_new_.element_count() > 0;
    if (is_varlen_k_new) {
      FFI_CHECK(cu_seqlens_k_new_.element_type() == ffi::DataType::S32)
          << "cu_seqlens_k_new must have dtype torch.int32";
    }
    int seqlen_k_new = !is_varlen_k_new ? k_new_.dimensions()[1] : 0;
    params.seqlen_knew = seqlen_k_new;
    params.total_knew = k_new_.dimensions()[0];
    params.knew_ptr = k_new_.untyped_data();
    params.vnew_ptr = v_new_.untyped_data();
    auto knew_s =
        cute::compact_row_major(cute::make_shape(batch_size, seqlen_k_new, num_heads_k, head_size));
    auto vnew_s = cute::compact_row_major(
        cute::make_shape(batch_size, seqlen_k_new, num_heads_k, head_size_v));

    params.knew_row_stride = cute::get<1>(knew_s);
    params.vnew_row_stride = cute::get<1>(vnew_s);
    params.knew_head_stride = cute::get<2>(knew_s);
    params.vnew_head_stride = cute::get<2>(vnew_s);
    if (!is_varlen_k_new) {
      params.knew_batch_stride = cute::get<0>(knew_s);
      params.vnew_batch_stride = cute::get<0>(vnew_s);
    }
    if (is_varlen_k_new) {
      params.cu_seqlens_knew = cu_seqlens_k_new_.typed_data<int>();
    }
  }
  bool const use_prepare_varlen = is_varlen;
  params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;
  // Temporarily set num_splits_dynamic_ptr to a non-null value since get_num_splits checks it.
  params.num_splits_dynamic_ptr = !use_prepare_varlen ? nullptr : reinterpret_cast<int *>(1);

  params.pagedkv_tma = get_pagedkv_tma(params);
  params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
  params.pack_gqa = pack_gqa_has_value ? pack_gqa_val : get_pack_gqa(params);

  // This needs to be set after get_num_splits
  void *tile_count_semaphore_ptr = nullptr;
  bool const scheduler_needs_semaphore =
      params.arch >= 90
          ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
          : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
  params.varlen_sort_batches = !params.is_local;
  params.head_swizzle = params.is_causal || params.is_local;

  if (scheduler_needs_semaphore || use_prepare_varlen) {
    int b_rounded = round_multiple(params.b, 4);
    int num_prepare_batch_vectors = use_prepare_varlen ? 2 : 0;
    if (params.varlen_sort_batches) {
      num_prepare_batch_vectors += 1;
    }
    if (params.head_swizzle) {
      num_prepare_batch_vectors += 1;
    }
    int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
    int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
    int metadata_size = int(scheduler_needs_semaphore) + tile_count_semaphore_offset;

    params.skip_scheduler_metadata_computation = (scheduler_metadata_.element_count() > 0);
    if (scheduler_metadata_.element_count() > 0) {
      FFI_CHECK(scheduler_metadata_.element_type() == ffi::DataType::S32);
      tile_count_semaphore_ptr = scheduler_metadata_.untyped_data();

    } else {
      auto sm_buffer = scratch.Allocate(metadata_size * sizeof(int32_t));
      tile_count_semaphore_ptr = sm_buffer.value();
    }
    if (scheduler_needs_semaphore && !use_prepare_varlen) {
      FFI_CUDA_CHECK(
          cudaMemsetAsync(tile_count_semaphore_ptr, 0, metadata_size * sizeof(int32_t), stream));
    }

    int32_t *base_ptr = static_cast<int32_t *>(tile_count_semaphore_ptr);
    params.num_splits_dynamic_ptr = use_prepare_varlen ? base_ptr : nullptr;
    params.num_m_blocks_ptr = use_prepare_varlen ? base_ptr + b_rounded : nullptr;
    params.varlen_batch_idx_ptr =
        use_prepare_varlen && params.varlen_sort_batches ? base_ptr + b_rounded * 2 : nullptr;
    params.num_nheads_in_l2_ptr =
        use_prepare_varlen && params.head_swizzle ? base_ptr + head_swizzle_offset : nullptr;
    params.tile_count_semaphore =
        scheduler_needs_semaphore ? base_ptr + tile_count_semaphore_offset : nullptr;
    params.tile_count_semaphore_offset = tile_count_semaphore_offset;
  }

  // Handle q_v for mixed-precision attention
  if (q_v_.element_count() > 0) {
    FFI_CHECK(head_size <= 64) << "q_v is only supported for head_size <= 64";
    FFI_CHECK(head_size_v >= 256) << "q_v is only supported for hdim_v >= 256.";
    FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
        << "q_v is only supported for fp16 and bf16";
    FFI_CHECK(params.arch == 90) << "q_v is only supported for Hopper GPUs";
    FFI_CHECK(q_v_.element_type() == dtype) << "q_v must have the same dtype as query";

    params.qv_ptr = q_v_.untyped_data();
    auto qv_s =
        cute::compact_row_major(cute::make_shape(batch_size, seqlen_q, num_heads, head_size_v));
    params.qv_row_stride = cute::get<1>(qv_s);
    params.qv_head_stride = cute::get<2>(qv_s);
    if (!is_varlen_q) {
      params.qv_batch_stride = cute::get<0>(qv_s);
    }
  }

  // Handle Rotary Positional Embeddings
  if (rotary_cos_.element_count() > 0) {
    FFI_CHECK(k_new_.element_count() > 0) << "Rotary embeddings require k_new_ to be provided";
    FFI_CHECK(rotary_sin_.element_count() > 0) << "rotary_sin must be provided with rotary_cos";
    FFI_CHECK(rotary_cos_.element_type() == dtype)
        << "rotary_cos must have the same dtype as query";
    FFI_CHECK(rotary_sin_.element_type() == dtype)
        << "rotary_sin must have the same dtype as query";

    params.rotary_dim = rotary_cos_.dimensions()[1] * 2;
    FFI_CHECK(params.rotary_dim <= head_size) << "rotary_dim must be <= headdim";
    FFI_CHECK(params.rotary_dim % 16 == 0) << "rotary_dim must be a multiple of 16";
    if (paged_KV) {
      FFI_CHECK(rotary_cos_.dimensions()[0] >= seqlen_k)
          << "cos/sin seqlen must be >= KV cache seqlen";
    }

    params.rotary_cos_ptr = rotary_cos_.untyped_data();
    params.rotary_sin_ptr = rotary_sin_.untyped_data();
    params.is_rotary_interleaved = is_rotary_interleaved;

    if (seqlens_rotary_.element_count() > 0) {
      FFI_CHECK(seqlens_rotary_.element_type() == ffi::DataType::S32)
          << "seqlens_rotary must be S32";
      params.seqlens_rotary = seqlens_rotary_.typed_data<int>();
    }

  } else {
    params.rotary_dim = 0;
  }

  if (kv_batch_idx_.element_count() > 0) {
    FFI_CHECK(kv_batch_idx_.element_type() == ffi::DataType::S32) << "kv_batch_idx must be S32";
    params.kv_batch_idx = kv_batch_idx_.typed_data<int>();
  }

  if (params.num_splits > 1) {
    FFI_CHECK(params.num_splits <= 256) << "num_splits > 256 not supported";
    FFI_CHECK(out_accum->element_count() > 0 && softmax_lse_accum->element_count() > 0)
        << "Accumulator buffers must be provided when num_splits > 1";

    params.oaccum_ptr = out_accum->untyped_data();
    params.softmax_lseaccum_ptr = softmax_lse_accum->untyped_data();
    params.is_fp32 = false;

    auto oaccum_s = cute::compact_row_major(
        cute::make_shape(params.num_splits, batch_size, num_heads, seqlen_q, head_size_v));
    auto lseaccum_s = cute::compact_row_major(
        cute::make_shape(params.num_splits, batch_size, num_heads, seqlen_q));
    params.oaccum_split_stride = cute::get<0>(oaccum_s);
    params.lseaccum_split_stride = cute::get<0>(lseaccum_s);

    if (!is_varlen_q) {
      params.oaccum_batch_stride = cute::get<1>(oaccum_s);
      params.lseaccum_batch_stride = cute::get<1>(lseaccum_s);
      params.oaccum_head_stride = cute::get<2>(oaccum_s);
      params.lseaccum_head_stride = cute::get<2>(lseaccum_s);
      params.oaccum_row_stride = cute::get<3>(oaccum_s);

    } else {
      // For varlen, the layout is different.
      auto oaccum_s_v = cute::compact_row_major(
          cute::make_shape(params.num_splits, num_heads, total_q, head_size_v));
      auto lseaccum_s_v =
          cute::compact_row_major(cute::make_shape(params.num_splits, num_heads, total_q));
      params.oaccum_head_stride = cute::get<1>(oaccum_s_v);
      params.lseaccum_head_stride = cute::get<1>(lseaccum_s_v);
      params.oaccum_row_stride = cute::get<2>(oaccum_s_v);
    }
  }

  if (dtype == ffi::DataType::F8E4M3FN) {
    if (q_descale_.untyped_data()) {
      params.q_descale_ptr = q_descale_.typed_data<float>();
      params.q_descale_batch_stride = q_descale_.dimensions()[1];
      params.q_descale_head_stride = 1;

    } else {
      params.q_descale_ptr = nullptr;
    }
    if (k_descale_.untyped_data()) {
      params.k_descale_ptr = k_descale_.typed_data<float>();
      params.k_descale_batch_stride = k_descale_.dimensions()[1];
      params.k_descale_head_stride = 1;

    } else {
      params.k_descale_ptr = nullptr;
    }
    if (v_descale_.untyped_data()) {
      params.v_descale_ptr = v_descale_.typed_data<float>();
      params.v_descale_batch_stride = v_descale_.dimensions()[1];
      params.v_descale_head_stride = 1;

    } else {
      params.v_descale_ptr = nullptr;
    }
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  FFI_CHECK(!params.is_local) << "This flash attention build does not support local attention.";
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  FFI_CHECK(params.softcap == 0.0)
      << "This flash attention build does not support tanh softcapping.";
#endif
#ifdef FLASHATTENTION_DISABLE_SPLIT
  FFI_CHECK(params.num_splits == 1) << "This flash attention build does not support splits.";
#endif
#ifdef FLASHATTENTION_DISABLE_PACKGQA
  FFI_CHECK(!params.pack_gqa || params.arch < 90 || (params.page_table && !params.pagedkv_tma) ||
            params.num_splits > 1)
      << "This build does not support pack_gqa.";
#endif
#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  FFI_CHECK(!(params.page_table && !params.pagedkv_tma)) << "This build does not support paged KV.";
#endif
#ifdef FLASHATTENTION_DISABLE_APPENDKV
  FFI_CHECK(k_new_.element_count() <= 0) << "This build does not support appending KV.";
#endif

  if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
    run_mha_fwd(params, stream);
    if (params.num_splits > 1) {
      if (out->element_type() == ffi::DataType::BF16) {
        params.is_bf16 = true;
      }
      run_mha_fwd_combine(params, stream, true /* enable_pdl */);

    } else if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation) {
      // If metadata was provided, we need to zero out the semaphore part for the next run.
      FFI_CUDA_CHECK(cudaMemsetAsync(params.tile_count_semaphore, 0, sizeof(int32_t), stream));
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    // Handle empty K/V cache: set output to 0 and LSE to infinity.
    FFI_CUDA_CHECK(cudaMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream));
    const size_t lse_num_elements = softmax_lse->size_bytes() / sizeof(float);
    if (lse_num_elements > 0) {
      fill_float(static_cast<float *>(softmax_lse->untyped_data()),
                 std::numeric_limits<float>::infinity(), lse_num_elements, stream);
      FFI_CUDA_CHECK(cudaGetLastError());
    }
  }

  return ffi::Error();
}

#ifdef FLASHATTENTION_DISABLE_BACKWARD
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  FFI_CHECK(false) << "Flash-Attention was built with backward disabled";
}
#else
template <int Arch, bool Has_softcap>
void run_mha_bwd_constexpr(Flash_bwd_params &params, cudaStream_t stream) {
  if (!params.is_bf16) {
#ifndef FLASHATTENTION_DISABLE_FP16
#ifndef FLASHATTENTION_DISABLE_HDIM64
    if (params.d_rounded == 64) {
      return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
    if (params.d_rounded == 96) {
      return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
    if (params.d_rounded == 128) {
      return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
    if (params.d_rounded == 192) {
      return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
    if (params.d_rounded == 256) {
      return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap>(params, stream);
    }
#endif
#else
    FFI_CHECK(false) << "This flash attention build does not support FP16.";
#endif
  } else {
#ifndef FLASHATTENTION_DISABLE_HDIM64
    if (params.d_rounded == 64) {
      return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
    if (params.d_rounded == 96) {
      return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
    if (params.d_rounded == 128) {
      return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
    if (params.d_rounded == 192) {
      return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
    if (params.d_rounded == 256) {
      return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap>(params, stream);
    }
#endif
  }
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  ARCH_SWITCH(params.arch, Arch, [&] {
    SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap,
                   [&] { run_mha_bwd_constexpr<Arch, Has_softcap>(params, stream); });
  });
}
#endif

ffi::Error
mha_bwd_impl(cudaStream_t stream, ffi::ScratchAllocator scratch, int32_t device,
             ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
             ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse, ffi::AnyBuffer cu_seqlens_q_,
             ffi::AnyBuffer cu_seqlens_k_, ffi::AnyBuffer seqused_q_, ffi::AnyBuffer seqused_k_,
             int64_t max_seqlen_q_val, bool max_seqlen_q_has_value, int64_t max_seqlen_k_val,
             bool max_seqlen_k_has_value, double softmax_scale_val, bool softmax_scale_has_value,
             bool is_causal, int64_t window_size_left, int64_t window_size_right, double softcap,
             bool deterministic, int64_t sm_margin, ffi::Result<ffi::AnyBuffer> dq,
             ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
             ffi::Result<ffi::AnyBuffer> softmax_d, ffi::Result<ffi::AnyBuffer> softmax_lse_log2) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  return ffi::Error(ffi::ErrorCode::kUnavailable,
                    "FlashAttention backward pass is disabled in this build.");
#endif

  int major, minor, sm_count;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  FFI_CHECK(major >= 8) << "FlashAttention only supports Ampere GPUs or newer.";
  auto arch = major * 10 + minor;

  auto dtype = q.element_type();
  FFI_CHECK(dtype == ffi::DataType::F16 || dtype == ffi::DataType::BF16)
      << "FlashAttention backward only supports fp16 and bf16 data types";
  FFI_CHECK(k.element_type() == dtype && v.element_type() == dtype && out.element_type() == dtype &&
            dout.element_type() == dtype)
      << "All input tensors must have the same dtype";

  const bool is_varlen_q = cu_seqlens_q_.element_count() > 0;
  if (is_varlen_q) {
    FFI_CHECK(cu_seqlens_q_.element_type() == ffi::DataType::S32)
        << "cu_seqlens_q must have dtype torch.int32";
    FFI_CHECK(max_seqlen_q_has_value) << "max_seqlen_q must be provided for varlen_q";
  }
  const bool is_varlen_k = cu_seqlens_k_.element_count() > 0;
  if (is_varlen_k) {
    FFI_CHECK(cu_seqlens_k_.element_type() == ffi::DataType::S32)
        << "cu_seqlens_k must have dtype torch.int32";
    FFI_CHECK(max_seqlen_k_has_value) << "max_seqlen_k must be provided for varlen_k";
  }
  const bool is_varlen = is_varlen_q || is_varlen_k || seqused_q_.element_count() > 0 ||
                         seqused_k_.element_count() > 0;
#ifdef FLASHATTENTION_DISABLE_VARLEN
  FFI_CHECK(!is_varlen) << "This flash attention build does not support varlen.";
#endif

  auto const sizes = q.dimensions();
  const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q_.dimensions()[0] - 1;
  const int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_val;
  const int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  // For varlen: q shape is [total_q, h, d], otherwise [b, s_q, h, d]
  const int num_heads = !is_varlen_q ? sizes[2] : sizes[1];
  const int head_size = !is_varlen_q ? sizes[3] : sizes[2];
  const int head_size_v = !is_varlen_q ? v.dimensions()[3] : v.dimensions()[2];
  const int seqlen_k = !is_varlen_k ? k.dimensions()[1] : max_seqlen_k_val;
  const int total_k = !is_varlen_k ? batch_size * k.dimensions()[1] : k.dimensions()[0];
  // For varlen: k shape is [total_k, h_k, d], otherwise [b, s_k, h_k, d]
  const int num_heads_k = !is_varlen_k ? k.dimensions()[2] : k.dimensions()[1];

  FFI_CHECK(head_size % 8 == 0 && head_size_v % 8 == 0) << "Head sizes must be multiples of 8";
  int const max_headdim = get_max_headdim();
  FFI_CHECK(std::max(head_size, head_size_v) <= max_headdim)
      << "FlashAttention forward only supports head dimension at most " +
             std::to_string(max_headdim);
  FFI_CHECK(num_heads % num_heads_k == 0)
      << "Number of heads in key/value must divide number of heads in query";

  double softmax_scale =
      softmax_scale_has_value ? softmax_scale_val : (1.0 / sqrt(static_cast<double>(head_size)));

  if (window_size_left >= seqlen_k - 1)
    window_size_left = -1;
  if (window_size_right >= seqlen_q - 1)
    window_size_right = -1;
  if (is_causal)
    window_size_right = 0;

  // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_bprop will set
  // params.is_causal=true. If we don't have is_causal here matching params.is_causal, we might get
  // the wrong kBlockM (and cause IMA).
  is_causal = window_size_left < 0 && window_size_right == 0;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded = round_up_headdim(std::max(head_size, head_size_v));
  int const head_size_v_rounded = head_size_rounded;
  FFI_CHECK(!deterministic || head_size_rounded < 256)
      << "Deterministic backward not supported for hdim 256.";
  // Very important that these match the kernel configs
  bool const is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
  int const kBlockM_sm90 =
      head_size_rounded <= 64
          ? (is_causal && softcap > 0.0 ? 96 : 128)
          : (head_size_rounded <= 96
                 ? 64
                 : (head_size_rounded <= 128 ? (is_causal || is_local || softcap > 0.0 ? 64 : 80)
                                             : 64));
  int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
  int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
  int const kBlockM =
      arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
  int const kBlockN_sm90 = head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 96 : 80);
  int const kBlockN_sm80 = head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 80 : 64);
  int const kBlockN_sm86 =
      head_size_rounded <= 64
          ? 128
          : (head_size_rounded <= 96
                 ? 128
                 : (head_size_rounded <= 128 ? 96 : (head_size_rounded <= 192 ? 64 : 64)));
  int const kBlockN =
      arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);

  const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
  const int seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
  const int total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
  const int total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);
  bool const seqused_q_has_value = seqused_q_.element_count() > 0;
  bool const seqused_k_has_value = seqused_k_.element_count() > 0;
  if (seqused_q_has_value) {
    FFI_CHECK(seqused_q_.element_type() == ffi::DataType::S32) << "seqused_q must have dtype int32";
  }
  if (seqused_k_has_value) {
    FFI_CHECK(seqused_k_.element_type() == ffi::DataType::S32) << "seqused_k must have dtype int32";
  }

  Flash_bwd_params params;
  set_params_dgrad(params, device, dtype, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
                   q.untyped_data(), k.untyped_data(), v.untyped_data(), out.untyped_data(),
                   dout.untyped_data(), dq->untyped_data(), dk->untyped_data(), dv->untyped_data(),
                   is_varlen_q ? cu_seqlens_q_.typed_data<int>() : nullptr,
                   is_varlen_k ? cu_seqlens_k_.typed_data<int>() : nullptr,
                   seqused_q_has_value ? seqused_q_.typed_data<int>() : nullptr,
                   seqused_k_has_value ? seqused_k_.typed_data<int>() : nullptr,
                   nullptr, // dq_accum_ptr,
                   nullptr, // dk_accum_ptr
                   nullptr, // dv_accum_ptr
                   softmax_lse.untyped_data(), softmax_d->untyped_data(), 0.f, softmax_scale,
                   window_size_left, window_size_right, 0, softcap, deterministic, sm_margin,
                   total_q, total_k);

  params.total_q = total_q;
  params.total_k = total_k;
  params.softmax_lse_log2_ptr = softmax_lse_log2->untyped_data();
  params.dv = head_size_v;
  params.dv_rounded = head_size_v_rounded;
  params.is_bf16 = (dtype == ffi::DataType::BF16);

  // Allocate all temporary buffers from the scratch allocator.
  size_t dq_accum_bytes =
      (!is_varlen ? (size_t)batch_size * num_heads * seqlen_q_rounded * head_size_rounded
                  : (size_t)num_heads * total_q_padded_rounded * head_size_rounded) *
      sizeof(float);
  auto dq_accum_buffer = scratch.Allocate(dq_accum_bytes);
  params.dq_accum_ptr = dq_accum_buffer.value();

  if (num_heads_k != num_heads) {
    size_t dk_accum_bytes =
        (!is_varlen ? (size_t)batch_size * num_heads_k * seqlen_k_rounded * head_size_rounded
                    : (size_t)num_heads_k * total_k_padded_rounded * head_size_rounded) *
        sizeof(float);
    auto dk_accum_buffer = scratch.Allocate(dk_accum_bytes);
    FFI_CUDA_CHECK(cudaMemsetAsync(dk_accum_buffer.value(), 0, dk_accum_bytes, stream));
    params.dk_accum_ptr = dk_accum_buffer.value();

    size_t dv_accum_bytes =
        (!is_varlen ? (size_t)batch_size * num_heads_k * seqlen_k_rounded * params.dv_rounded
                    : (size_t)num_heads_k * total_k_padded_rounded * params.dv_rounded) *
        sizeof(float);
    auto dv_accum_buffer = scratch.Allocate(dv_accum_bytes);
    FFI_CUDA_CHECK(cudaMemsetAsync(dv_accum_buffer.value(), 0, dv_accum_bytes, stream));
    params.dv_accum_ptr = dv_accum_buffer.value();
  }

  size_t dq_semaphore_bytes =
      (size_t)batch_size * num_heads * ((seqlen_q + kBlockM - 1) / kBlockM) * sizeof(int32_t);
  auto dq_semaphore_buffer = scratch.Allocate(dq_semaphore_bytes);
  params.dq_semaphore = static_cast<int *>(dq_semaphore_buffer.value());

  if (num_heads_k != num_heads && params.deterministic) {
    size_t dkv_semaphore_bytes =
        (size_t)batch_size * num_heads_k * ((seqlen_k + kBlockN - 1) / kBlockN) * sizeof(int32_t);
    auto dk_semaphore_buffer = scratch.Allocate(dkv_semaphore_bytes);
    auto dv_semaphore_buffer = scratch.Allocate(dkv_semaphore_bytes);
    FFI_CUDA_CHECK(cudaMemsetAsync(dk_semaphore_buffer.value(), 0, dkv_semaphore_bytes, stream));
    FFI_CUDA_CHECK(cudaMemsetAsync(dv_semaphore_buffer.value(), 0, dkv_semaphore_bytes, stream));
    params.dk_semaphore = static_cast<int *>(dk_semaphore_buffer.value());
    params.dv_semaphore = static_cast<int *>(dv_semaphore_buffer.value());
  }

  if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
    run_mha_bwd(params, stream);
  } else if (total_k > 0 && num_heads_k > 0) {
    FFI_CUDA_CHECK(cudaMemsetAsync(dk->untyped_data(), 0, dk->size_bytes(), stream));
    FFI_CUDA_CHECK(cudaMemsetAsync(dv->untyped_data(), 0, dv->size_bytes(), stream));
    FFI_CUDA_CHECK(cudaMemsetAsync(softmax_d->untyped_data(), 0, softmax_d->size_bytes(), stream));
  } else if (total_q > 0 && num_heads_k > 0) {
    FFI_CUDA_CHECK(cudaMemsetAsync(dq->untyped_data(), 0, dq->size_bytes(), stream));
    FFI_CUDA_CHECK(cudaMemsetAsync(softmax_d->untyped_data(), 0, softmax_d->size_bytes(), stream));
  }

  return ffi::Error::Success();
}

ffi::Error mha_combine_impl(cudaStream_t stream, int32_t device, ffi::AnyBuffer out_partial,
                            ffi::AnyBuffer lse_partial, int64_t head_size_og,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::AnyBuffer> softmax_lse) {
  int major, minor;
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  FFI_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  FFI_CHECK(major >= 8) << "Attention combine function only supports Ampere GPUs or newer.";

  FFI_CHECK(out_partial.element_type() == ffi::DataType::F32 &&
            lse_partial.element_type() == ffi::DataType::F32)
      << "Partial accumulators for combine must be of F32 type";

  const auto sizes = out_partial.dimensions();
  const int num_splits = sizes[0];
  const int batch_size = sizes[1];
  const int seqlen = sizes[2];
  const int num_heads = sizes[3];
  const int head_size = sizes[4];

  FFI_CHECK(num_splits <= 256) << "Combine only supports num_splits <= 256";
  FFI_CHECK(head_size % 4 == 0) << "Padded head_size must be a multiple of 4";

  Flash_fwd_params params{};
  params.is_fp32 = (out->element_type() == ffi::DataType::F32);
  params.is_bf16 = (out->element_type() == ffi::DataType::BF16);

  params.oaccum_ptr = out_partial.untyped_data();
  params.softmax_lseaccum_ptr = lse_partial.untyped_data();
  params.o_ptr = out->untyped_data();
  params.softmax_lse_ptr = softmax_lse->untyped_data();

  params.b = batch_size;
  params.h = num_heads;
  params.seqlen_q = seqlen;
  params.dv = head_size;
  params.num_splits = num_splits;

  auto oaccum_s = cute::compact_row_major(
      cute::make_shape(num_splits, batch_size, seqlen, num_heads, head_size));
  auto lseaccum_s =
      cute::compact_row_major(cute::make_shape(num_splits, batch_size, seqlen, num_heads));
  auto out_s = cute::compact_row_major(cute::make_shape(batch_size, seqlen, num_heads, head_size));

  params.oaccum_split_stride = cute::get<0>(oaccum_s);
  params.oaccum_batch_stride = cute::get<1>(oaccum_s);
  params.oaccum_row_stride = cute::get<2>(oaccum_s);
  params.oaccum_head_stride = cute::get<3>(oaccum_s);

  params.lseaccum_split_stride = cute::get<0>(lseaccum_s);
  params.lseaccum_batch_stride = cute::get<1>(lseaccum_s);
  params.lseaccum_head_stride = cute::get<3>(lseaccum_s);

  params.o_batch_stride = cute::get<0>(out_s);
  params.o_row_stride = cute::get<1>(out_s);
  params.o_head_stride = cute::get<2>(out_s);

  params.arch = major * 10 + minor;

  if (seqlen > 0 && batch_size > 0) {
    run_mha_fwd_combine(params, stream, false /*enable_pdl*/);
  }

  return ffi::Error::Success();
}
