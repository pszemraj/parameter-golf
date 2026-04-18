// HGDN block megakernel candidate.
//
// This is the first repo-backed, architecture-faithful parity kernel:
// - packed dense W_qkv
// - packed depthwise causal qkv conv + SiLU
// - q/k L2 normalization
// - fp32 gate math
// - fp32 gated-delta recurrence state
// - fp32 output RMSNorm
// - SiLU output gate
// - dense cross-head W_out
//
// Local `sm_89` or `sm_120` runs are correctness-only. H100 inference from
// those timings is explicitly invalid.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {
using bf16 = __nv_bfloat16;

constexpr int MAX_D = 512;
constexpr int MAX_H = 16;
constexpr int MAX_DK = 64;
constexpr int MAX_DV = 64;
constexpr int MAX_K = 8;
constexpr int WARP_SIZE = 32;
#ifndef HGDN_THREADS
#define HGDN_THREADS 128
#endif
constexpr int THREADS = HGDN_THREADS;
static_assert(
    THREADS % WARP_SIZE == 0,
    "HGDN megakernel requires THREADS to be a multiple of warp size.");
constexpr int WARPS_PER_BLOCK = THREADS / WARP_SIZE;
constexpr int GEMM_TILE = 16;
#ifndef HGDN_GEMM_ATB_SPLIT_M_THRESHOLD
#define HGDN_GEMM_ATB_SPLIT_M_THRESHOLD 2048
#endif
constexpr int GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD =
    HGDN_GEMM_ATB_SPLIT_M_THRESHOLD;
constexpr int REC_DOT_COL_SLICE = 8;
#ifndef HGDN_REC_V_TILE
#define HGDN_REC_V_TILE 8
#endif

#ifndef HGDN_REC_CHUNK_T
#define HGDN_REC_CHUNK_T 8
#endif

constexpr int REC_V_TILE = HGDN_REC_V_TILE;
constexpr int REC_CHUNK_T_MAX = HGDN_REC_CHUNK_T;
constexpr float EPS = 1.0e-6f;

struct DeviceReport {
  int major;
  int minor;
  int multiProcessorCount;
  int l2CacheSize;
  int regsPerMultiprocessor;
  int warpSize;
  int maxThreadsPerMultiProcessor;
  int cooperativeLaunch;
  size_t sharedMemPerMultiprocessor;
  size_t sharedMemPerBlockOptin;
  std::string name;
};

DeviceReport report() {
  int device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  return {
      props.major,
      props.minor,
      props.multiProcessorCount,
      props.l2CacheSize,
      props.regsPerMultiprocessor,
      props.warpSize,
      props.maxThreadsPerMultiProcessor,
      props.cooperativeLaunch,
      props.sharedMemPerMultiprocessor,
      props.sharedMemPerBlockOptin,
      props.name,
  };
}

std::string device_report_string() {
  DeviceReport r = report();
  std::ostringstream out;
  out << "GPU_NAME=" << r.name << "\n";
  out << "major=" << r.major << "\n";
  out << "minor=" << r.minor << "\n";
  out << "multiProcessorCount=" << r.multiProcessorCount << "\n";
  out << "l2CacheSize=" << r.l2CacheSize << "\n";
  out << "sharedMemPerMultiprocessor=" << r.sharedMemPerMultiprocessor << "\n";
  out << "sharedMemPerBlockOptin=" << r.sharedMemPerBlockOptin << "\n";
  out << "regsPerMultiprocessor=" << r.regsPerMultiprocessor << "\n";
  out << "warpSize=" << r.warpSize << "\n";
  out << "maxThreadsPerMultiProcessor=" << r.maxThreadsPerMultiProcessor << "\n";
  out << "cooperativeLaunch=" << r.cooperativeLaunch << "\n";
  out << "scope="
      << ((r.major == 8 && r.minor == 9) || (r.major == 12 && r.minor == 0)
              ? "correctness_only_non_h100"
              : (r.major == 9 ? "target_h100_class" : "non_h100_unknown"))
      << "\n";
  return out.str();
}

void validate_runtime_device() {
  DeviceReport r = report();
  TORCH_CHECK(
      r.cooperativeLaunch,
      "HGDN megakernel requires cooperative launch support on the active CUDA device.");
  TORCH_CHECK(
      r.sharedMemPerBlockOptin >= 48 * 1024,
      "HGDN megakernel requires enough opt-in shared memory for the parity kernel.");
}

inline void chk_bf16(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16, name, " must be bfloat16");
}

inline void chk_f32(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
}

inline bf16* bptr(torch::Tensor& t) {
  return reinterpret_cast<bf16*>(t.data_ptr<c10::BFloat16>());
}

inline const bf16* cbptr(const torch::Tensor& t) {
  return reinterpret_cast<const bf16*>(t.data_ptr<c10::BFloat16>());
}

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float sig(float x) {
  return 1.0f / (1.0f + expf(-x));
}
__device__ __forceinline__ float softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return expf(x);
  }
  return log1pf(expf(x));
}
__device__ __forceinline__ float silu(float x) { return x * sig(x); }
__device__ __forceinline__ float dsilu(float x) {
  float s = sig(x);
  return s * (1.0f + x * (1.0f - s));
}

__device__ __forceinline__ float warp_sum(float x) {
  unsigned mask = 0xffffffffu;
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(mask, x, offset);
  }
  return x;
}

__device__ __forceinline__ float block_sum(float x, float* scratch) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  x = warp_sum(x);
  if (lane == 0) {
    scratch[warp_id] = x;
  }
  __syncthreads();
  if (warp_id == 0) {
    float total = lane < WARPS_PER_BLOCK ? scratch[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) {
      scratch[0] = total;
    }
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ void block_sum2(float& x, float& y, float* scratch) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  x = warp_sum(x);
  y = warp_sum(y);
  if (lane == 0) {
    scratch[warp_id] = x;
    scratch[WARPS_PER_BLOCK + warp_id] = y;
  }
  __syncthreads();
  if (warp_id == 0) {
    float total_x = lane < WARPS_PER_BLOCK ? scratch[lane] : 0.0f;
    float total_y =
        lane < WARPS_PER_BLOCK ? scratch[WARPS_PER_BLOCK + lane] : 0.0f;
    total_x = warp_sum(total_x);
    total_y = warp_sum(total_y);
    if (lane == 0) {
      scratch[0] = total_x;
      scratch[1] = total_y;
    }
  }
  __syncthreads();
  x = scratch[0];
  y = scratch[1];
}

__device__ __forceinline__ void block_dot_cols_slice8(
    const float* mat,
    const float* vec,
    float* out,
    int Dk,
    int stride,
    int cols,
    float* scratch,
    float scale = 1.0f) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;
  int col = lane >> 2;
  int subgroup_lane = lane & 3;
  float acc = 0.0f;
  for (int i = warp_id * 4 + subgroup_lane; i < Dk;
       i += 4 * WARPS_PER_BLOCK) {
    if (col < cols) {
      acc += mat[i * stride + col] * vec[i];
    }
  }
  acc = __shfl_down_sync(0xffffffffu, acc, 2, 4) + acc;
  acc = __shfl_down_sync(0xffffffffu, acc, 1, 4) + acc;
  if (subgroup_lane == 0 && col < cols) {
    scratch[warp_id * REC_DOT_COL_SLICE + col] = acc * scale;
  }
  __syncthreads();
  if (threadIdx.x < cols) {
    float total = 0.0f;
    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
      total += scratch[w * REC_DOT_COL_SLICE + threadIdx.x];
    }
    out[threadIdx.x] = total;
  }
  __syncthreads();
}

__device__ __forceinline__ void block_dot_cols_tiled(
    const float* mat,
    const float* vec,
    float* out,
    int Dk,
    int stride,
    int cols,
    float* scratch,
    float scale = 1.0f) {
  for (int col_base = 0; col_base < cols; col_base += REC_DOT_COL_SLICE) {
    int cols_this_slice = min(REC_DOT_COL_SLICE, cols - col_base);
    block_dot_cols_slice8(
        mat + col_base,
        vec,
        out + col_base,
        Dk,
        stride,
        cols_this_slice,
        scratch,
        scale);
  }
}

__device__ __forceinline__ int64_t idx3(
    int b,
    int t,
    int c,
    int T,
    int C) {
  return (static_cast<int64_t>(b) * T + t) * C + c;
}

__device__ __forceinline__ int64_t idx4(
    int b,
    int t,
    int h,
    int d,
    int T,
    int H,
    int D) {
  return ((static_cast<int64_t>(b) * T + t) * H + h) * D + d;
}

__device__ __forceinline__ int64_t checkpoint_idx(
    int b,
    int chunk,
    int h,
    int i,
    int j,
    int NChunks,
    int H,
    int Dk,
    int Dv) {
  return ((((static_cast<int64_t>(b) * NChunks + chunk) * H + h) * Dk + i) *
          Dv +
          j);
}

__device__ __forceinline__ int cq(int h, int d, int Dk) { return h * Dk + d; }
__device__ __forceinline__ int ck(int h, int d, int H, int Dk) {
  return H * Dk + h * Dk + d;
}
__device__ __forceinline__ int cv(int h, int d, int H, int Dk, int Dv) {
  return 2 * H * Dk + h * Dv + d;
}

__device__ float dot_row(
    const bf16* x,
    const bf16* w,
    int row,
    int b,
    int t,
    int D,
    int T) {
  float acc = 0.0f;
  int64_t x_base = idx3(b, t, 0, T, D);
  int64_t w_base = static_cast<int64_t>(row) * D;
  for (int d = 0; d < D; ++d) {
    acc += b2f(x[x_base + d]) * b2f(w[w_base + d]);
  }
  return acc;
}

__device__ __forceinline__ int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

template <typename StoreFn>
__device__ void phase_gemm_warp_scalar(
    int m0,
    int n0,
    int M,
    int N,
    int lane,
    StoreFn&& store_value) {
  for (int idx = lane; idx < GEMM_TILE * GEMM_TILE; idx += WARP_SIZE) {
    int r = idx / GEMM_TILE;
    int c = idx % GEMM_TILE;
    int m = m0 + r;
    int n = n0 + c;
    if (m < M && n < N) {
      store_value(m, n);
    }
  }
}

__device__ void phase_gemm_abt_store_bf16(
    const bf16* A,
    const bf16* B,
    bf16* Out,
    int M,
    int K,
    int N,
    float* shmem) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int warp_stride = gridDim.x * WARPS_PER_BLOCK;
  int tiles_n = ceil_div_int(N, GEMM_TILE);
  int total_tiles = ceil_div_int(M, GEMM_TILE) * tiles_n;
  float* warp_tile = shmem + warp_id * GEMM_TILE * GEMM_TILE;

  for (int tile = global_warp; tile < total_tiles; tile += warp_stride) {
    int tile_m = tile / tiles_n;
    int tile_n = tile % tiles_n;
    int m0 = tile_m * GEMM_TILE;
    int n0 = tile_n * GEMM_TILE;
    bool full_tile = (m0 + GEMM_TILE <= M) && (n0 + GEMM_TILE <= N) &&
                     (K % GEMM_TILE == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::col_major>
          b_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          float>
          acc_frag;
      nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
      for (int k0 = 0; k0 < K; k0 += GEMM_TILE) {
        const bf16* a_ptr = A + static_cast<int64_t>(m0) * K + k0;
        const bf16* b_ptr = B + static_cast<int64_t>(n0) * K + k0;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, K);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, K);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      nvcuda::wmma::store_matrix_sync(
          warp_tile, acc_frag, GEMM_TILE, nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < GEMM_TILE * GEMM_TILE; idx += WARP_SIZE) {
        int r = idx / GEMM_TILE;
        int c = idx % GEMM_TILE;
        Out[static_cast<int64_t>(m0 + r) * N + (n0 + c)] = f2b(warp_tile[idx]);
      }
      __syncwarp();
      continue;
    }
#endif

    phase_gemm_warp_scalar(
        m0, n0, M, N, lane, [&](int m, int n) {
          float acc = 0.0f;
          int64_t a_base = static_cast<int64_t>(m) * K;
          int64_t b_base = static_cast<int64_t>(n) * K;
          for (int k = 0; k < K; ++k) {
            acc += b2f(A[a_base + k]) * b2f(B[b_base + k]);
          }
          Out[static_cast<int64_t>(m) * N + n] = f2b(acc);
        });
  }
}

__device__ void phase_gemm_ab_store_bf16(
    const bf16* A,
    const bf16* B,
    bf16* Out,
    int M,
    int K,
    int N,
    float* shmem) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int warp_stride = gridDim.x * WARPS_PER_BLOCK;
  int tiles_n = ceil_div_int(N, GEMM_TILE);
  int total_tiles = ceil_div_int(M, GEMM_TILE) * tiles_n;
  float* warp_tile = shmem + warp_id * GEMM_TILE * GEMM_TILE;

  for (int tile = global_warp; tile < total_tiles; tile += warp_stride) {
    int tile_m = tile / tiles_n;
    int tile_n = tile % tiles_n;
    int m0 = tile_m * GEMM_TILE;
    int n0 = tile_n * GEMM_TILE;
    bool full_tile = (m0 + GEMM_TILE <= M) && (n0 + GEMM_TILE <= N) &&
                     (K % GEMM_TILE == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          float>
          acc_frag;
      nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
      for (int k0 = 0; k0 < K; k0 += GEMM_TILE) {
        const bf16* a_ptr = A + static_cast<int64_t>(m0) * K + k0;
        const bf16* b_ptr = B + static_cast<int64_t>(k0) * N + n0;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, K);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, N);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      nvcuda::wmma::store_matrix_sync(
          warp_tile, acc_frag, GEMM_TILE, nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < GEMM_TILE * GEMM_TILE; idx += WARP_SIZE) {
        int r = idx / GEMM_TILE;
        int c = idx % GEMM_TILE;
        Out[static_cast<int64_t>(m0 + r) * N + (n0 + c)] = f2b(warp_tile[idx]);
      }
      __syncwarp();
      continue;
    }
#endif

    phase_gemm_warp_scalar(
        m0, n0, M, N, lane, [&](int m, int n) {
          float acc = 0.0f;
          int64_t a_base = static_cast<int64_t>(m) * K;
          for (int k = 0; k < K; ++k) {
            acc += b2f(A[a_base + k]) * b2f(B[static_cast<int64_t>(k) * N + n]);
          }
          Out[static_cast<int64_t>(m) * N + n] = f2b(acc);
        });
  }
}

__device__ void phase_gemm_ab_accum_f32(
    const bf16* A,
    const bf16* B,
    float* Out,
    int M,
    int K,
    int N,
    float* shmem,
    bool zero_init) {
  (void)shmem;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int warp_stride = gridDim.x * WARPS_PER_BLOCK;
  int tiles_n = ceil_div_int(N, GEMM_TILE);
  int total_tiles = ceil_div_int(M, GEMM_TILE) * tiles_n;
  for (int tile = global_warp; tile < total_tiles; tile += warp_stride) {
    int tile_m = tile / tiles_n;
    int tile_n = tile % tiles_n;
    int m0 = tile_m * GEMM_TILE;
    int n0 = tile_n * GEMM_TILE;
    bool full_tile = (m0 + GEMM_TILE <= M) && (n0 + GEMM_TILE <= N) &&
                     (K % GEMM_TILE == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          float>
          acc_frag;
      if (zero_init) {
        nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
      } else {
        nvcuda::wmma::load_matrix_sync(
            acc_frag,
            Out + static_cast<int64_t>(m0) * N + n0,
            N,
            nvcuda::wmma::mem_row_major);
      }
      for (int k0 = 0; k0 < K; k0 += GEMM_TILE) {
        const bf16* a_ptr = A + static_cast<int64_t>(m0) * K + k0;
        const bf16* b_ptr = B + static_cast<int64_t>(k0) * N + n0;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, K);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, N);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      nvcuda::wmma::store_matrix_sync(
          Out + static_cast<int64_t>(m0) * N + n0,
          acc_frag,
          N,
          nvcuda::wmma::mem_row_major);
      continue;
    }
#endif

    phase_gemm_warp_scalar(
        m0, n0, M, N, lane, [&](int m, int n) {
          float acc = zero_init ? 0.0f : Out[static_cast<int64_t>(m) * N + n];
          int64_t a_base = static_cast<int64_t>(m) * K;
          for (int k = 0; k < K; ++k) {
            acc += b2f(A[a_base + k]) * b2f(B[static_cast<int64_t>(k) * N + n]);
          }
          Out[static_cast<int64_t>(m) * N + n] = acc;
        });
  }
}

__device__ void phase_gemm_aT_b_store_bf16(
    const bf16* A,
    const bf16* B,
    bf16* Out,
    int M,
    int R,
    int C,
    float* shmem) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int warp_stride = gridDim.x * WARPS_PER_BLOCK;
  int tiles_c = ceil_div_int(C, GEMM_TILE);
  int total_tiles = ceil_div_int(R, GEMM_TILE) * tiles_c;
  float* warp_tile = shmem + warp_id * GEMM_TILE * GEMM_TILE;

  for (int tile = global_warp; tile < total_tiles; tile += warp_stride) {
    int tile_r = tile / tiles_c;
    int tile_c = tile % tiles_c;
    int r0 = tile_r * GEMM_TILE;
    int c0 = tile_c * GEMM_TILE;
    bool full_tile = (r0 + GEMM_TILE <= R) && (c0 + GEMM_TILE <= C) &&
                     (M % GEMM_TILE == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::col_major>
          a_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          float>
          acc_frag;
      nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
      for (int m0 = 0; m0 < M; m0 += GEMM_TILE) {
        const bf16* a_ptr = A + static_cast<int64_t>(m0) * R + r0;
        const bf16* b_ptr = B + static_cast<int64_t>(m0) * C + c0;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, R);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, C);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      nvcuda::wmma::store_matrix_sync(
          warp_tile, acc_frag, GEMM_TILE, nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int idx = lane; idx < GEMM_TILE * GEMM_TILE; idx += WARP_SIZE) {
        int r = idx / GEMM_TILE;
        int c = idx % GEMM_TILE;
        Out[static_cast<int64_t>(r0 + r) * C + (c0 + c)] = f2b(warp_tile[idx]);
      }
      __syncwarp();
      continue;
    }
#endif

    phase_gemm_warp_scalar(
        r0, c0, R, C, lane, [&](int r, int c) {
          float acc = 0.0f;
          for (int m = 0; m < M; ++m) {
            acc += b2f(A[static_cast<int64_t>(m) * R + r]) *
                   b2f(B[static_cast<int64_t>(m) * C + c]);
          }
          Out[static_cast<int64_t>(r) * C + c] = f2b(acc);
        });
  }
}

__device__ void phase_gemm_aT_b_store_bf16_splitm_block(
    const bf16* A,
    const bf16* B,
    bf16* Out,
    int M,
    int R,
    int C,
    float* shmem) {
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int tiles_c = ceil_div_int(C, GEMM_TILE);
  int total_tiles = ceil_div_int(R, GEMM_TILE) * tiles_c;
  float* warp_tile = shmem + warp_id * GEMM_TILE * GEMM_TILE;

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    int tile_r = tile / tiles_c;
    int tile_c = tile % tiles_c;
    int r0 = tile_r * GEMM_TILE;
    int c0 = tile_c * GEMM_TILE;
    bool full_tile = (r0 + GEMM_TILE <= R) && (c0 + GEMM_TILE <= C) &&
                     (M % GEMM_TILE == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_tile) {
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::col_major>
          a_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          bf16,
          nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          GEMM_TILE,
          GEMM_TILE,
          GEMM_TILE,
          float>
          acc_frag;
      nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
      for (int m0 = warp_id * GEMM_TILE; m0 < M;
           m0 += WARPS_PER_BLOCK * GEMM_TILE) {
        const bf16* a_ptr = A + static_cast<int64_t>(m0) * R + r0;
        const bf16* b_ptr = B + static_cast<int64_t>(m0) * C + c0;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, R);
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, C);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      nvcuda::wmma::store_matrix_sync(
          warp_tile, acc_frag, GEMM_TILE, nvcuda::wmma::mem_row_major);
      __syncthreads();
      for (int idx = threadIdx.x; idx < GEMM_TILE * GEMM_TILE;
           idx += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
          acc += shmem[w * GEMM_TILE * GEMM_TILE + idx];
        }
        int r = idx / GEMM_TILE;
        int c = idx % GEMM_TILE;
        Out[static_cast<int64_t>(r0 + r) * C + (c0 + c)] = f2b(acc);
      }
      __syncthreads();
      continue;
    }
#endif

    for (int idx = threadIdx.x;
         idx < WARPS_PER_BLOCK * GEMM_TILE * GEMM_TILE;
         idx += blockDim.x) {
      shmem[idx] = 0.0f;
    }
    __syncthreads();
    phase_gemm_warp_scalar(
        r0, c0, R, C, lane, [&](int r, int c) {
          float acc = 0.0f;
          for (int m = warp_id; m < M; m += WARPS_PER_BLOCK) {
            acc += b2f(A[static_cast<int64_t>(m) * R + r]) *
                   b2f(B[static_cast<int64_t>(m) * C + c]);
          }
          warp_tile[(r - r0) * GEMM_TILE + (c - c0)] = acc;
        });
    __syncthreads();
    for (int idx = threadIdx.x; idx < GEMM_TILE * GEMM_TILE;
         idx += blockDim.x) {
      int r = idx / GEMM_TILE;
      int c = idx % GEMM_TILE;
      int rr = r0 + r;
      int cc = c0 + c;
      if (rr < R && cc < C) {
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
          acc += shmem[w * GEMM_TILE * GEMM_TILE + idx];
        }
        Out[static_cast<int64_t>(rr) * C + cc] = f2b(acc);
      }
    }
    __syncthreads();
  }
}

__device__ float conv_at(
    const bf16* packed,
    const bf16* weight,
    int b,
    int t,
    int c,
    int T,
    int C,
    int K) {
  float acc = 0.0f;
  int start = t - (K - 1);
  int64_t w_base = static_cast<int64_t>(c) * K;
  for (int tap = 0; tap < K; ++tap) {
    int tau = start + tap;
    if (tau >= 0 && tau < T) {
      acc += b2f(packed[idx3(b, tau, c, T, C)]) * b2f(weight[w_base + tap]);
    }
  }
  return acc;
}

struct ForwardSharedState {
  float* S0;
  float* S1;
  float* q;
  float* k;
  float* v;
  float* tmp_dv0;
  float* scratch;
};

struct BackwardSharedState {
  float* S0;
  float* S1;
  float* adj;
  float* q;
  float* k;
  float* v;
  float* go;
  float* tmp_dv0;
  float* tmp_dv1;
  float* tmp_dk0;
  float* reduce0;
  float* q_hist;
  float* k_hist;
  float* v_hist;
  float* alpha_hist;
  float* beta_hist;
};

__host__ __device__ __forceinline__ size_t forward_recurrence_bytes(
    int Dk,
    int Dv) {
  int dv_tile = Dv < REC_V_TILE ? Dv : REC_V_TILE;
  return static_cast<size_t>(
             2 * Dk * dv_tile + 2 * Dk + 2 * dv_tile +
             WARPS_PER_BLOCK * REC_V_TILE) *
         sizeof(float);
}

__host__ __device__ __forceinline__ size_t backward_recurrence_bytes(
    int Dk,
    int Dv) {
  int dv_tile = Dv < REC_V_TILE ? Dv : REC_V_TILE;
  return static_cast<size_t>(
             3 * Dk * dv_tile + 3 * Dk + 4 * dv_tile + THREADS +
             REC_CHUNK_T_MAX * (2 * Dk + dv_tile + 2)) *
         sizeof(float);
}

__device__ ForwardSharedState forward_shared_ptrs(float* base, int Dk, int Dv) {
  ForwardSharedState s{};
  int offset = 0;
  s.S0 = base + offset;
  offset += Dk * Dv;
  s.S1 = base + offset;
  offset += Dk * Dv;
  s.q = base + offset;
  offset += Dk;
  s.k = base + offset;
  offset += Dk;
  s.v = base + offset;
  offset += Dv;
  s.tmp_dv0 = base + offset;
  offset += Dv;
  s.scratch = base + offset;
  return s;
}

__device__ BackwardSharedState backward_shared_ptrs(
    float* base,
    int Dk,
    int Dv) {
  BackwardSharedState s{};
  int offset = 0;
  s.S0 = base + offset;
  offset += Dk * Dv;
  s.S1 = base + offset;
  offset += Dk * Dv;
  s.adj = base + offset;
  offset += Dk * Dv;
  s.q = base + offset;
  offset += Dk;
  s.k = base + offset;
  offset += Dk;
  s.v = base + offset;
  offset += Dv;
  s.go = base + offset;
  offset += Dv;
  s.tmp_dv0 = base + offset;
  offset += Dv;
  s.tmp_dv1 = base + offset;
  offset += Dv;
  s.tmp_dk0 = base + offset;
  offset += Dk;
  s.reduce0 = base + offset;
  offset += THREADS;
  s.q_hist = base + offset;
  offset += REC_CHUNK_T_MAX * Dk;
  s.k_hist = base + offset;
  offset += REC_CHUNK_T_MAX * Dk;
  s.v_hist = base + offset;
  offset += REC_CHUNK_T_MAX * Dv;
  s.alpha_hist = base + offset;
  offset += REC_CHUNK_T_MAX;
  s.beta_hist = base + offset;
  return s;
}

__global__ void hgdn_forward_bf16_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w_qkv,
    const bf16* __restrict__ w_a,
    const bf16* __restrict__ w_b,
    const bf16* __restrict__ w_g,
    const bf16* __restrict__ w_out,
    const bf16* __restrict__ conv_w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    bf16* __restrict__ y,
    bf16* __restrict__ qkv,
    bf16* __restrict__ pre_tmp,
    bf16* __restrict__ g_pre,
    bf16* __restrict__ beta_pre,
    bf16* __restrict__ g_log,
    bf16* __restrict__ beta,
    bf16* __restrict__ g_out,
    bf16* __restrict__ o_raw,
    bf16* __restrict__ z_tmp,
    float* __restrict__ state_ckpt,
    int B,
    int T,
    int D,
    int H,
    int Dk,
    int Dv,
    int K,
    int n_chunks,
    int rec_chunk_t,
    int allow_neg_eigval) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float shmem[];

  int Cq = H * Dk;
  int P = H * Dv;
  int C = 2 * Cq + P;
  int64_t BT = static_cast<int64_t>(B) * T;
  int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t linear_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  phase_gemm_abt_store_bf16(x, w_qkv, qkv, static_cast<int>(BT), D, C, shmem);
  grid.sync();

  for (int64_t idx = linear_tid; idx < BT * H; idx += linear_stride) {
    int h = idx % H;
    int64_t bt = idx / H;
    int t = bt % T;
    int b = bt / T;
    g_pre[idx] = f2b(dot_row(x, w_a, h, b, t, D, T));
    beta_pre[idx] = f2b(dot_row(x, w_b, h, b, t, D, T));
  }
  phase_gemm_abt_store_bf16(x, w_g, g_out, static_cast<int>(BT), D, P, shmem);
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
      int q_channel = cq(h, d, Dk);
      int k_channel = ck(h, d, H, Dk);
      float q_preact = conv_at(qkv, conv_w, b, t, q_channel, T, C, K);
      float k_preact = conv_at(qkv, conv_w, b, t, k_channel, T, C, K);
      pre_tmp[idx3(b, t, q_channel, T, C)] = f2b(q_preact);
      pre_tmp[idx3(b, t, k_channel, T, C)] = f2b(k_preact);
    }
    for (int d = threadIdx.x; d < Dv; d += blockDim.x) {
      int v_channel = cv(h, d, H, Dk, Dv);
      float v_preact = conv_at(qkv, conv_w, b, t, v_channel, T, C, K);
      pre_tmp[idx3(b, t, v_channel, T, C)] = f2b(v_preact);
    }
    __syncthreads();
  }
  grid.sync();

  for (int64_t idx = linear_tid; idx < BT * H; idx += linear_stride) {
    int h = idx % H;
    float exp_A = expf(A_log[h]);
    float g_value = -exp_A * softplus(b2f(g_pre[idx]) + dt_bias[h]);
    float beta_value = sig(b2f(beta_pre[idx]));
    if (allow_neg_eigval) {
      beta_value *= 2.0f;
    }
    g_log[idx] = f2b(g_value);
    beta[idx] = f2b(beta_value);
  }
  grid.sync();

  int v_tiles = ceil_div_int(Dv, REC_V_TILE);
  for (int job = blockIdx.x; job < B * H * v_tiles; job += gridDim.x) {
    int tile = job % v_tiles;
    int head_job = job / v_tiles;
    int b = head_job / H;
    int h = head_job % H;
    int v_base = tile * REC_V_TILE;
    int dv_local = min(REC_V_TILE, Dv - v_base);
    ForwardSharedState shared = forward_shared_ptrs(shmem, Dk, dv_local);
    int state_elems = Dk * dv_local;

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      shared.S0[idx] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
      float sq = 0.0f;
      float sk = 0.0f;
      for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
        int q_channel = cq(h, i, Dk);
        int k_channel = ck(h, i, H, Dk);
        float q_value = silu(b2f(pre_tmp[idx3(b, t, q_channel, T, C)]));
        float k_value = silu(b2f(pre_tmp[idx3(b, t, k_channel, T, C)]));
        shared.q[i] = q_value;
        shared.k[i] = k_value;
        sq += q_value * q_value;
        sk += k_value * k_value;
      }
      for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
        int v_channel = cv(h, v_base + j, H, Dk, Dv);
        shared.v[j] = silu(b2f(pre_tmp[idx3(b, t, v_channel, T, C)]));
      }
      block_sum2(sq, sk, shared.scratch);
      float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
      float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
      for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
        shared.q[i] *= inv_q_norm;
        shared.k[i] *= inv_k_norm;
      }
      __syncthreads();

      if ((t % rec_chunk_t) == 0) {
        int chunk = t / rec_chunk_t;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          state_ckpt[checkpoint_idx(
              b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)] =
              shared.S0[idx];
        }
      }
      block_dot_cols_tiled(
          shared.S0,
          shared.k,
          shared.tmp_dv0,
          Dk,
          dv_local,
          dv_local,
          shared.scratch);
      __syncthreads();

      float alpha = expf(b2f(g_log[idx3(b, t, h, T, H)]));
      float beta_value = b2f(beta[idx3(b, t, h, T, H)]);
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        int i = idx / dv_local;
        int j = idx - i * dv_local;
        shared.S1[idx] = alpha *
                             (shared.S0[idx] -
                              beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                         beta_value * shared.k[i] * shared.v[j];
      }
      __syncthreads();
      block_dot_cols_tiled(
          shared.S1,
          shared.q,
          shared.v,
          Dk,
          dv_local,
          dv_local,
          shared.scratch);
      for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
        o_raw[idx4(b, t, h, v_base + j, T, H, Dv)] = f2b(shared.v[j]);
      }
      __syncthreads();
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        shared.S0[idx] = shared.S1[idx];
      }
      __syncthreads();
    }
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      sum_sq += o_value * o_value;
    }
    sum_sq = block_sum(sum_sq, shmem);
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(Dv) + EPS);
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      int p = h * Dv + j;
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]) * inv_rms;
      float gate_value = silu(b2f(g_out[idx3(b, t, p, T, P)]));
      z_tmp[idx3(b, t, p, T, P)] = f2b(o_value * gate_value);
    }
    __syncthreads();
  }
  grid.sync();

  phase_gemm_abt_store_bf16(
      z_tmp, w_out, y, static_cast<int>(BT), P, D, shmem);
}

__global__ void hgdn_backward_bf16_kernel(
    const bf16* __restrict__ grad_y,
    const bf16* __restrict__ x,
    const bf16* __restrict__ w_qkv,
    const bf16* __restrict__ w_a,
    const bf16* __restrict__ w_b,
    const bf16* __restrict__ w_g,
    const bf16* __restrict__ w_out,
    const bf16* __restrict__ conv_w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const bf16* __restrict__ qkv,
    const bf16* __restrict__ g_pre,
    const bf16* __restrict__ beta_pre,
    const bf16* __restrict__ g_log,
    const bf16* __restrict__ beta,
    const bf16* __restrict__ g_out,
    const bf16* __restrict__ o_raw,
    const float* __restrict__ state_ckpt,
    float* __restrict__ grad_q_norm_accum,
    float* __restrict__ grad_k_norm_accum,
    float* __restrict__ grad_g_log_accum,
    float* __restrict__ grad_beta_accum,
    float* __restrict__ grad_x_accum,
    bf16* __restrict__ grad_x,
    bf16* __restrict__ grad_w_qkv,
    bf16* __restrict__ grad_w_a,
    bf16* __restrict__ grad_w_b,
    bf16* __restrict__ grad_w_g,
    bf16* __restrict__ grad_w_out,
    bf16* __restrict__ grad_conv_w,
    float* __restrict__ grad_A_log,
    float* __restrict__ grad_dt_bias,
    bf16* __restrict__ grad_z,
    bf16* __restrict__ grad_o_raw,
    bf16* __restrict__ grad_g_out,
    bf16* __restrict__ grad_v_post,
    bf16* __restrict__ grad_pre,
    bf16* __restrict__ grad_qkv,
    bf16* __restrict__ grad_g_pre,
    bf16* __restrict__ grad_beta_pre,
    int B,
    int T,
    int D,
    int H,
    int Dk,
    int Dv,
    int K,
    int n_chunks,
    int rec_chunk_t,
    int allow_neg_eigval) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float shmem[];

  int Cq = H * Dk;
  int P = H * Dv;
  int C = 2 * Cq + P;
  int64_t BT = static_cast<int64_t>(B) * T;
  int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t linear_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = linear_tid; idx < BT * D; idx += linear_stride) {
    grad_x_accum[idx] = 0.0f;
  }
  for (int64_t idx = linear_tid; idx < BT * H * Dk; idx += linear_stride) {
    grad_q_norm_accum[idx] = 0.0f;
    grad_k_norm_accum[idx] = 0.0f;
  }
  for (int64_t idx = linear_tid; idx < BT * H; idx += linear_stride) {
    grad_g_log_accum[idx] = 0.0f;
    grad_beta_accum[idx] = 0.0f;
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      sum_sq += o_value * o_value;
    }
    sum_sq = block_sum(sum_sq, shmem);
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(Dv) + EPS);
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      int p = h * Dv + j;
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]) * inv_rms;
      float gate_value = silu(b2f(g_out[idx3(b, t, p, T, P)]));
      grad_z[idx3(b, t, p, T, P)] = f2b(o_value * gate_value);
    }
    __syncthreads();
  }
  grid.sync();
  phase_gemm_aT_b_store_bf16(
      grad_y, grad_z, grad_w_out, static_cast<int>(BT), D, P, shmem);
  grid.sync();
  phase_gemm_ab_store_bf16(
      grad_y, w_out, grad_z, static_cast<int>(BT), D, P, shmem);
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      sum_sq += o_value * o_value;
    }
    sum_sq = block_sum(sum_sq, shmem);
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(Dv) + EPS);
    float dot_go = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      int p = h * Dv + j;
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      float on_value = o_value * inv_rms;
      float gate_input = b2f(g_out[idx3(b, t, p, T, P)]);
      float dz_value = b2f(grad_z[idx3(b, t, p, T, P)]);
      float grad_norm = dz_value * silu(gate_input);
      grad_g_out[idx3(b, t, p, T, P)] =
          f2b(dz_value * on_value * dsilu(gate_input));
      dot_go += grad_norm * o_value;
      grad_o_raw[idx4(b, t, h, j, T, H, Dv)] = f2b(grad_norm);
    }
    dot_go = block_sum(dot_go, shmem);
    float coeff =
        inv_rms * inv_rms * inv_rms * dot_go / static_cast<float>(Dv);
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      float grad_norm = b2f(grad_o_raw[idx4(b, t, h, j, T, H, Dv)]);
      grad_o_raw[idx4(b, t, h, j, T, H, Dv)] =
          f2b(inv_rms * grad_norm - coeff * o_value);
    }
    __syncthreads();
  }
  grid.sync();

  int v_tiles = ceil_div_int(Dv, REC_V_TILE);
  for (int job = blockIdx.x; job < B * H * v_tiles; job += gridDim.x) {
    int tile = job % v_tiles;
    int head_job = job / v_tiles;
    int b = head_job / H;
    int h = head_job % H;
    int v_base = tile * REC_V_TILE;
    int dv_local = min(REC_V_TILE, Dv - v_base);
    BackwardSharedState shared = backward_shared_ptrs(shmem, Dk, dv_local);
    int state_elems = Dk * dv_local;

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      shared.adj[idx] = 0.0f;
    }
    __syncthreads();

    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
      int chunk_start = chunk * rec_chunk_t;
      int chunk_end = min(T, chunk_start + rec_chunk_t);
      int chunk_len = chunk_end - chunk_start;

      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        int i = idx / dv_local;
        int j = idx - i * dv_local;
        shared.S0[idx] = state_ckpt[checkpoint_idx(
            b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)];
      }
      __syncthreads();

      for (int local_t = 0; local_t < chunk_len; ++local_t) {
        int t = chunk_start + local_t;
        float sq = 0.0f;
        float sk = 0.0f;
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          int q_channel = cq(h, i, Dk);
          int k_channel = ck(h, i, H, Dk);
          float q_preact = conv_at(qkv, conv_w, b, t, q_channel, T, C, K);
          float k_preact = conv_at(qkv, conv_w, b, t, k_channel, T, C, K);
          float q_activated = silu(q_preact);
          float k_activated = silu(k_preact);
          shared.q[i] = q_activated;
          shared.k[i] = k_activated;
          sq += q_activated * q_activated;
          sk += k_activated * k_activated;
        }
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          int v_channel = cv(h, v_base + j, H, Dk, Dv);
          float v_value = silu(conv_at(qkv, conv_w, b, t, v_channel, T, C, K));
          shared.v[j] = v_value;
          shared.v_hist[local_t * dv_local + j] = v_value;
        }
        block_sum2(sq, sk, shared.reduce0);
        float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
        float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float q_value = shared.q[i] * inv_q_norm;
          float k_value = shared.k[i] * inv_k_norm;
          shared.q[i] = q_value;
          shared.k[i] = k_value;
          shared.q_hist[local_t * Dk + i] = q_value;
          shared.k_hist[local_t * Dk + i] = k_value;
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.S0,
            shared.k,
            shared.tmp_dv0,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        __syncthreads();

        float alpha = expf(b2f(g_log[idx3(b, t, h, T, H)]));
        float beta_value = b2f(beta[idx3(b, t, h, T, H)]);
        if (threadIdx.x == 0) {
          shared.alpha_hist[local_t] = alpha;
          shared.beta_hist[local_t] = beta_value;
        }
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha *
                               (shared.S0[idx] -
                                beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                           beta_value * shared.k[i] * shared.v[j];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          shared.S0[idx] = shared.S1[idx];
        }
        __syncthreads();
      }

      for (int local_t = chunk_len - 1; local_t >= 0; --local_t) {
        int t = chunk_start + local_t;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S0[idx] = state_ckpt[checkpoint_idx(
              b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)];
        }
        __syncthreads();

        for (int replay_t = 0; replay_t < local_t; ++replay_t) {
          for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
            shared.k[i] = shared.k_hist[replay_t * Dk + i];
          }
          for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
            shared.v[j] = shared.v_hist[replay_t * dv_local + j];
          }
          __syncthreads();

          block_dot_cols_tiled(
              shared.S0,
              shared.k,
              shared.tmp_dv0,
              Dk,
              dv_local,
              dv_local,
              shared.reduce0);
          __syncthreads();

          float replay_alpha = shared.alpha_hist[replay_t];
          float replay_beta = shared.beta_hist[replay_t];
          for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
            int i = idx / dv_local;
            int j = idx - i * dv_local;
            shared.S1[idx] = replay_alpha *
                                 (shared.S0[idx] -
                                  replay_beta * shared.k[i] * shared.tmp_dv0[j]) +
                             replay_beta * shared.k[i] * shared.v[j];
          }
          __syncthreads();
          for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
            shared.S0[idx] = shared.S1[idx];
          }
          __syncthreads();
        }

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          shared.q[i] = shared.q_hist[local_t * Dk + i];
          shared.k[i] = shared.k_hist[local_t * Dk + i];
        }
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          shared.v[j] = shared.v_hist[local_t * dv_local + j];
          shared.go[j] = b2f(grad_o_raw[idx4(b, t, h, v_base + j, T, H, Dv)]);
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.S0,
            shared.k,
            shared.tmp_dv0,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        __syncthreads();

        float alpha = shared.alpha_hist[local_t];
        float beta_value = shared.beta_hist[local_t];
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha *
                               (shared.S0[idx] -
                                beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                           beta_value * shared.k[i] * shared.v[j];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float acc = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            acc += shared.S1[i * dv_local + j] * shared.go[j];
          }
          atomicAdd(&grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)], acc);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.adj[idx] += shared.q[i] * shared.go[j];
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.adj,
            shared.k,
            shared.tmp_dv1,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0,
            alpha);
        __syncthreads();

        block_dot_cols_tiled(
            shared.adj,
            shared.k,
            shared.go,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          grad_v_post[idx4(b, t, h, v_base + j, T, H, Dv)] =
              f2b(beta_value * shared.go[j]);
        }
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float acc = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            acc += shared.adj[i * dv_local + j] * shared.v[j];
          }
          shared.tmp_dk0[i] = acc;
        }

        float local_da = 0.0f;
        float local_db = 0.0f;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          float Aij =
              shared.S0[idx] - beta_value * shared.k[i] * shared.tmp_dv0[j];
          local_da += shared.adj[idx] * Aij;
          local_db +=
              (alpha * shared.adj[idx]) * (-shared.k[i] * shared.tmp_dv0[j]) +
              shared.adj[idx] * shared.k[i] * shared.v[j];
        }
        block_sum2(local_da, local_db, shared.reduce0);
        if (threadIdx.x == 0) {
          atomicAdd(
              &grad_g_log_accum[idx3(b, t, h, T, H)],
              local_da * alpha);
          atomicAdd(
              &grad_beta_accum[idx3(b, t, h, T, H)],
              local_db);
        }

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float term0 = shared.tmp_dk0[i];
          float term1 = 0.0f;
          float term2 = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            term1 += (alpha * shared.adj[i * dv_local + j]) * shared.tmp_dv0[j];
            term2 += shared.S0[i * dv_local + j] * shared.tmp_dv1[j];
          }
          atomicAdd(
              &grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)],
              beta_value * (term0 - term1 - term2));
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha * shared.adj[idx] -
                           beta_value * shared.k[i] * shared.tmp_dv1[j];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          shared.adj[idx] = shared.S1[idx];
        }
        __syncthreads();
      }
    }
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float* q_preact_s = shmem;
    float* k_preact_s = shmem + Dk;
    float* reduce_s = shmem + 2 * Dk;
    float dot_q = 0.0f;
    float dot_k = 0.0f;
    float sq = 0.0f;
    float sk = 0.0f;
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      int q_channel = cq(h, i, Dk);
      int k_channel = ck(h, i, H, Dk);
      float q_preact = conv_at(qkv, conv_w, b, t, q_channel, T, C, K);
      float k_preact = conv_at(qkv, conv_w, b, t, k_channel, T, C, K);
      q_preact_s[i] = q_preact;
      k_preact_s[i] = k_preact;
      float q_value = silu(q_preact);
      float k_value = silu(k_preact);
      sq += q_value * q_value;
      sk += k_value * k_value;
    }
    block_sum2(sq, sk, reduce_s);

    float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
    float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      float dq = grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float dk = grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float q_value = silu(q_preact_s[i]) * inv_q_norm;
      float k_value = silu(k_preact_s[i]) * inv_k_norm;
      dot_q += dq * q_value;
      dot_k += dk * k_value;
    }
    block_sum2(dot_q, dot_k, reduce_s);
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      int q_channel = cq(h, i, Dk);
      int k_channel = ck(h, i, H, Dk);
      float dq = grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float dk = grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float q_preact = q_preact_s[i];
      float k_preact = k_preact_s[i];
      float q_value = silu(q_preact) * inv_q_norm;
      float k_value = silu(k_preact) * inv_k_norm;
      grad_pre[idx3(b, t, q_channel, T, C)] =
          f2b((dq - q_value * dot_q) * inv_q_norm * dsilu(q_preact));
      grad_pre[idx3(b, t, k_channel, T, C)] =
          f2b((dk - k_value * dot_k) * inv_k_norm * dsilu(k_preact));
    }
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      int v_channel = cv(h, j, H, Dk, Dv);
      float v_preact = conv_at(qkv, conv_w, b, t, v_channel, T, C, K);
      float grad_value = b2f(grad_v_post[idx4(b, t, h, j, T, H, Dv)]);
      grad_pre[idx3(b, t, v_channel, T, C)] =
          f2b(grad_value * dsilu(v_preact));
    }
    __syncthreads();
  }
  grid.sync();

  for (int64_t idx = linear_tid; idx < BT * C; idx += linear_stride) {
    int c = idx % C;
    int64_t bt = idx / C;
    int tau = bt % T;
    int b = bt / T;
    float acc = 0.0f;
    for (int tap = 0; tap < K; ++tap) {
      int t = tau + (K - 1) - tap;
      if (t >= 0 && t < T) {
        acc += b2f(grad_pre[idx3(b, t, c, T, C)]) *
               b2f(conv_w[static_cast<int64_t>(c) * K + tap]);
      }
    }
    grad_qkv[idx] = f2b(acc);
  }
  for (int64_t idx = linear_tid; idx < static_cast<int64_t>(C) * K;
       idx += linear_stride) {
    int tap = idx % K;
    int c = idx / K;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < T; ++t) {
        int tau = t - (K - 1) + tap;
        if (tau >= 0 && tau < T) {
          acc += b2f(grad_pre[idx3(b, t, c, T, C)]) *
                 b2f(qkv[idx3(b, tau, c, T, C)]);
        }
      }
    }
    grad_conv_w[idx] = f2b(acc);
  }
  grid.sync();

  for (int64_t h = linear_tid; h < H; h += linear_stride) {
    grad_A_log[h] = 0.0f;
    grad_dt_bias[h] = 0.0f;
  }
  grid.sync();

  float beta_scale = allow_neg_eigval ? 2.0f : 1.0f;
  for (int h = 0; h < H; ++h) {
    float local_grad_A = 0.0f;
    float local_grad_dt = 0.0f;
    float exp_A = expf(A_log[h]);
    for (int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         bt < BT;
         bt += linear_stride) {
      int64_t idx = bt * H + h;
      float z_value = b2f(g_pre[idx]) + dt_bias[h];
      float sig_z = sig(z_value);
      float grad_g_log_value = grad_g_log_accum[idx];
      float grad_dt_value = grad_g_log_value * (-exp_A * sig_z);
      grad_g_pre[idx] = f2b(grad_dt_value);
      float s = sig(b2f(beta_pre[idx]));
      grad_beta_pre[idx] =
          f2b(grad_beta_accum[idx] * beta_scale * s * (1.0f - s));
      local_grad_A += grad_g_log_value * (-exp_A * softplus(z_value));
      local_grad_dt += grad_dt_value;
    }
    block_sum2(local_grad_A, local_grad_dt, shmem);
    if (threadIdx.x == 0) {
      atomicAdd(&grad_A_log[h], local_grad_A);
      atomicAdd(&grad_dt_bias[h], local_grad_dt);
    }
    __syncthreads();
  }
  grid.sync();

  phase_gemm_ab_accum_f32(
      grad_qkv, w_qkv, grad_x_accum, static_cast<int>(BT), C, D, shmem, true);
  grid.sync();
  if (BT >= GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD) {
    phase_gemm_aT_b_store_bf16_splitm_block(
        grad_qkv, x, grad_w_qkv, static_cast<int>(BT), C, D, shmem);
  } else {
    phase_gemm_aT_b_store_bf16(
        grad_qkv, x, grad_w_qkv, static_cast<int>(BT), C, D, shmem);
  }
  grid.sync();
  for (int64_t idx = linear_tid; idx < static_cast<int64_t>(H) * D;
       idx += linear_stride) {
    int d = idx % D;
    int h = idx / D;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (int64_t bt = 0; bt < BT; ++bt) {
      acc_a += b2f(grad_g_pre[bt * H + h]) * b2f(x[bt * D + d]);
      acc_b += b2f(grad_beta_pre[bt * H + h]) * b2f(x[bt * D + d]);
    }
    grad_w_a[idx] = f2b(acc_a);
    grad_w_b[idx] = f2b(acc_b);
  }
  grid.sync();
  phase_gemm_ab_accum_f32(
      grad_g_out, w_g, grad_x_accum, static_cast<int>(BT), P, D, shmem, false);
  grid.sync();
  phase_gemm_aT_b_store_bf16(
      grad_g_out, x, grad_w_g, static_cast<int>(BT), P, D, shmem);
  grid.sync();
  for (int64_t idx = linear_tid; idx < BT * D; idx += linear_stride) {
    int d = idx % D;
    int64_t bt = idx / D;
    float acc = grad_x_accum[idx];
    for (int h = 0; h < H; ++h) {
      acc += b2f(grad_g_pre[bt * H + h]) *
             b2f(w_a[static_cast<int64_t>(h) * D + d]);
      acc += b2f(grad_beta_pre[bt * H + h]) *
             b2f(w_b[static_cast<int64_t>(h) * D + d]);
    }
    grad_x[idx] = f2b(acc);
  }
}

__global__ void hgdn_core_forward_bf16_kernel(
    const bf16* __restrict__ qkv,
    const bf16* __restrict__ g_pre,
    const bf16* __restrict__ beta_pre,
    const bf16* __restrict__ g_out,
    const bf16* __restrict__ conv_w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    bf16* __restrict__ z,
    bf16* __restrict__ g_log,
    bf16* __restrict__ beta,
    bf16* __restrict__ o_raw,
    float* __restrict__ state_ckpt,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int K,
    int n_chunks,
    int rec_chunk_t,
    int allow_neg_eigval) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float shmem[];

  int C = H * (2 * Dk + Dv);
  int64_t BT = static_cast<int64_t>(B) * T;
  int64_t linear_tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t linear_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = linear_tid; idx < BT * H; idx += linear_stride) {
    int h = idx % H;
    float exp_A = expf(A_log[h]);
    float g_value = -exp_A * softplus(b2f(g_pre[idx]) + dt_bias[h]);
    float beta_value = sig(b2f(beta_pre[idx]));
    if (allow_neg_eigval) {
      beta_value *= 2.0f;
    }
    g_log[idx] = f2b(g_value);
    beta[idx] = f2b(beta_value);
  }
  grid.sync();

  int v_tiles = ceil_div_int(Dv, REC_V_TILE);
  for (int job = blockIdx.x; job < B * H * v_tiles; job += gridDim.x) {
    int tile = job % v_tiles;
    int head_job = job / v_tiles;
    int b = head_job / H;
    int h = head_job % H;
    int v_base = tile * REC_V_TILE;
    int dv_local = min(REC_V_TILE, Dv - v_base);
    ForwardSharedState shared = forward_shared_ptrs(shmem, Dk, dv_local);
    int state_elems = Dk * dv_local;

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      shared.S0[idx] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
      float sq = 0.0f;
      float sk = 0.0f;
      for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
        int q_channel = cq(h, i, Dk);
        int k_channel = ck(h, i, H, Dk);
        float q_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, q_channel, T, C, K)));
        float k_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, k_channel, T, C, K)));
        float q_value = silu(q_preact);
        float k_value = silu(k_preact);
        shared.q[i] = q_value;
        shared.k[i] = k_value;
        sq += q_value * q_value;
        sk += k_value * k_value;
      }
      for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
        int v_channel = cv(h, v_base + j, H, Dk, Dv);
        float v_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, v_channel, T, C, K)));
        shared.v[j] = silu(v_preact);
      }
      block_sum2(sq, sk, shared.scratch);
      float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
      float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
      for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
        shared.q[i] *= inv_q_norm;
        shared.k[i] *= inv_k_norm;
      }
      __syncthreads();

      if ((t % rec_chunk_t) == 0) {
        int chunk = t / rec_chunk_t;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          state_ckpt[checkpoint_idx(
              b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)] =
              shared.S0[idx];
        }
      }
      block_dot_cols_tiled(
          shared.S0,
          shared.k,
          shared.tmp_dv0,
          Dk,
          dv_local,
          dv_local,
          shared.scratch);
      __syncthreads();

      float alpha = expf(b2f(g_log[idx3(b, t, h, T, H)]));
      float beta_value = b2f(beta[idx3(b, t, h, T, H)]);
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        int i = idx / dv_local;
        int j = idx - i * dv_local;
        shared.S1[idx] = alpha *
                             (shared.S0[idx] -
                              beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                         beta_value * shared.k[i] * shared.v[j];
      }
      __syncthreads();
      block_dot_cols_tiled(
          shared.S1,
          shared.q,
          shared.v,
          Dk,
          dv_local,
          dv_local,
          shared.scratch);
      for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
        o_raw[idx4(b, t, h, v_base + j, T, H, Dv)] = f2b(shared.v[j]);
      }
      __syncthreads();
      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        shared.S0[idx] = shared.S1[idx];
      }
      __syncthreads();
    }
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      sum_sq += o_value * o_value;
    }
    sum_sq = block_sum(sum_sq, shmem);
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(Dv) + EPS);
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]) * inv_rms;
      float gate_value = silu(b2f(g_out[idx4(b, t, h, j, T, H, Dv)]));
      z[idx4(b, t, h, j, T, H, Dv)] = f2b(o_value * gate_value);
    }
    __syncthreads();
  }
}

__global__ void hgdn_core_backward_bf16_kernel(
    const bf16* __restrict__ grad_z,
    const bf16* __restrict__ qkv,
    const bf16* __restrict__ g_pre,
    const bf16* __restrict__ beta_pre,
    const bf16* __restrict__ g_out,
    const bf16* __restrict__ conv_w,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const bf16* __restrict__ g_log,
    const bf16* __restrict__ beta,
    const bf16* __restrict__ o_raw,
    const float* __restrict__ state_ckpt,
    float* __restrict__ grad_q_norm_accum,
    float* __restrict__ grad_k_norm_accum,
    float* __restrict__ grad_g_log_accum,
    float* __restrict__ grad_beta_accum,
    bf16* __restrict__ grad_g_out,
    bf16* __restrict__ grad_o_raw,
    bf16* __restrict__ grad_v_post,
    bf16* __restrict__ grad_pre,
    bf16* __restrict__ grad_qkv,
    bf16* __restrict__ grad_g_pre,
    bf16* __restrict__ grad_beta_pre,
    bf16* __restrict__ grad_conv_w,
    float* __restrict__ grad_A_log,
    float* __restrict__ grad_dt_bias,
    int B,
    int T,
    int H,
    int Dk,
    int Dv,
    int K,
    int n_chunks,
    int rec_chunk_t,
    int allow_neg_eigval) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ float shmem[];

  int C = H * (2 * Dk + Dv);
  int64_t BT = static_cast<int64_t>(B) * T;
  int64_t linear_tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t linear_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = linear_tid; idx < BT * H * Dk; idx += linear_stride) {
    grad_q_norm_accum[idx] = 0.0f;
    grad_k_norm_accum[idx] = 0.0f;
  }
  for (int64_t idx = linear_tid; idx < BT * H; idx += linear_stride) {
    grad_g_log_accum[idx] = 0.0f;
    grad_beta_accum[idx] = 0.0f;
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float sum_sq = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      sum_sq += o_value * o_value;
    }
    sum_sq = block_sum(sum_sq, shmem);
    float inv_rms = rsqrtf(sum_sq / static_cast<float>(Dv) + EPS);
    float dot_go = 0.0f;
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      float on_value = o_value * inv_rms;
      float gate_input = b2f(g_out[idx4(b, t, h, j, T, H, Dv)]);
      float dz_value = b2f(grad_z[idx4(b, t, h, j, T, H, Dv)]);
      float grad_norm = dz_value * silu(gate_input);
      grad_g_out[idx4(b, t, h, j, T, H, Dv)] =
          f2b(dz_value * on_value * dsilu(gate_input));
      dot_go += grad_norm * o_value;
      grad_o_raw[idx4(b, t, h, j, T, H, Dv)] = f2b(grad_norm);
    }
    dot_go = block_sum(dot_go, shmem);
    float coeff =
        inv_rms * inv_rms * inv_rms * dot_go / static_cast<float>(Dv);
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      float o_value = b2f(o_raw[idx4(b, t, h, j, T, H, Dv)]);
      float grad_norm = b2f(grad_o_raw[idx4(b, t, h, j, T, H, Dv)]);
      grad_o_raw[idx4(b, t, h, j, T, H, Dv)] =
          f2b(inv_rms * grad_norm - coeff * o_value);
    }
    __syncthreads();
  }
  grid.sync();

  int v_tiles = ceil_div_int(Dv, REC_V_TILE);
  for (int job = blockIdx.x; job < B * H * v_tiles; job += gridDim.x) {
    int tile = job % v_tiles;
    int head_job = job / v_tiles;
    int b = head_job / H;
    int h = head_job % H;
    int v_base = tile * REC_V_TILE;
    int dv_local = min(REC_V_TILE, Dv - v_base);
    BackwardSharedState shared = backward_shared_ptrs(shmem, Dk, dv_local);
    int state_elems = Dk * dv_local;

    for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      shared.adj[idx] = 0.0f;
    }
    __syncthreads();

    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
      int chunk_start = chunk * rec_chunk_t;
      int chunk_end = min(T, chunk_start + rec_chunk_t);
      int chunk_len = chunk_end - chunk_start;

      for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
        int i = idx / dv_local;
        int j = idx - i * dv_local;
        shared.S0[idx] = state_ckpt[checkpoint_idx(
            b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)];
      }
      __syncthreads();

      for (int local_t = 0; local_t < chunk_len; ++local_t) {
        int t = chunk_start + local_t;
        float sq = 0.0f;
        float sk = 0.0f;
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          int q_channel = cq(h, i, Dk);
          int k_channel = ck(h, i, H, Dk);
          float q_preact =
              b2f(f2b(conv_at(qkv, conv_w, b, t, q_channel, T, C, K)));
          float k_preact =
              b2f(f2b(conv_at(qkv, conv_w, b, t, k_channel, T, C, K)));
          float q_activated = silu(q_preact);
          float k_activated = silu(k_preact);
          shared.q[i] = q_activated;
          shared.k[i] = k_activated;
          sq += q_activated * q_activated;
          sk += k_activated * k_activated;
        }
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          int v_channel = cv(h, v_base + j, H, Dk, Dv);
          float v_preact =
              b2f(f2b(conv_at(qkv, conv_w, b, t, v_channel, T, C, K)));
          float v_value = silu(v_preact);
          shared.v[j] = v_value;
          shared.v_hist[local_t * dv_local + j] = v_value;
        }
        block_sum2(sq, sk, shared.reduce0);
        float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
        float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float q_value = shared.q[i] * inv_q_norm;
          float k_value = shared.k[i] * inv_k_norm;
          shared.q[i] = q_value;
          shared.k[i] = k_value;
          shared.q_hist[local_t * Dk + i] = q_value;
          shared.k_hist[local_t * Dk + i] = k_value;
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.S0,
            shared.k,
            shared.tmp_dv0,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        __syncthreads();

        float alpha = expf(b2f(g_log[idx3(b, t, h, T, H)]));
        float beta_value = b2f(beta[idx3(b, t, h, T, H)]);
        if (threadIdx.x == 0) {
          shared.alpha_hist[local_t] = alpha;
          shared.beta_hist[local_t] = beta_value;
        }
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha *
                               (shared.S0[idx] -
                                beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                           beta_value * shared.k[i] * shared.v[j];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          shared.S0[idx] = shared.S1[idx];
        }
        __syncthreads();
      }

      for (int local_t = chunk_len - 1; local_t >= 0; --local_t) {
        int t = chunk_start + local_t;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S0[idx] = state_ckpt[checkpoint_idx(
              b, chunk, h, i, v_base + j, n_chunks, H, Dk, Dv)];
        }
        __syncthreads();

        for (int replay_t = 0; replay_t < local_t; ++replay_t) {
          for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
            shared.k[i] = shared.k_hist[replay_t * Dk + i];
          }
          for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
            shared.v[j] = shared.v_hist[replay_t * dv_local + j];
          }
          __syncthreads();

          block_dot_cols_tiled(
              shared.S0,
              shared.k,
              shared.tmp_dv0,
              Dk,
              dv_local,
              dv_local,
              shared.reduce0);
          __syncthreads();

          float replay_alpha = shared.alpha_hist[replay_t];
          float replay_beta = shared.beta_hist[replay_t];
          for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
            int i = idx / dv_local;
            int j = idx - i * dv_local;
            shared.S1[idx] = replay_alpha *
                                 (shared.S0[idx] -
                                  replay_beta * shared.k[i] * shared.tmp_dv0[j]) +
                             replay_beta * shared.k[i] * shared.v[j];
          }
          __syncthreads();
          for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
            shared.S0[idx] = shared.S1[idx];
          }
          __syncthreads();
        }

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          shared.q[i] = shared.q_hist[local_t * Dk + i];
          shared.k[i] = shared.k_hist[local_t * Dk + i];
        }
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          shared.v[j] = shared.v_hist[local_t * dv_local + j];
          shared.go[j] = b2f(grad_o_raw[idx4(b, t, h, v_base + j, T, H, Dv)]);
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.S0,
            shared.k,
            shared.tmp_dv0,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        __syncthreads();

        float alpha = shared.alpha_hist[local_t];
        float beta_value = shared.beta_hist[local_t];
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha *
                               (shared.S0[idx] -
                                beta_value * shared.k[i] * shared.tmp_dv0[j]) +
                           beta_value * shared.k[i] * shared.v[j];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float acc = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            acc += shared.S1[i * dv_local + j] * shared.go[j];
          }
          atomicAdd(&grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)], acc);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.adj[idx] += shared.q[i] * shared.go[j];
        }
        __syncthreads();

        block_dot_cols_tiled(
            shared.adj,
            shared.k,
            shared.tmp_dv1,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0,
            alpha);
        __syncthreads();

        block_dot_cols_tiled(
            shared.adj,
            shared.k,
            shared.go,
            Dk,
            dv_local,
            dv_local,
            shared.reduce0);
        for (int j = threadIdx.x; j < dv_local; j += blockDim.x) {
          grad_v_post[idx4(b, t, h, v_base + j, T, H, Dv)] =
              f2b(beta_value * shared.go[j]);
        }
        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float acc = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            acc += shared.adj[i * dv_local + j] * shared.v[j];
          }
          shared.tmp_dk0[i] = acc;
        }

        float local_da = 0.0f;
        float local_db = 0.0f;
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          float Aij =
              shared.S0[idx] - beta_value * shared.k[i] * shared.tmp_dv0[j];
          local_da += shared.adj[idx] * Aij;
          local_db +=
              (alpha * shared.adj[idx]) * (-shared.k[i] * shared.tmp_dv0[j]) +
              shared.adj[idx] * shared.k[i] * shared.v[j];
        }
        block_sum2(local_da, local_db, shared.reduce0);
        if (threadIdx.x == 0) {
          atomicAdd(&grad_g_log_accum[idx3(b, t, h, T, H)], local_da * alpha);
          atomicAdd(&grad_beta_accum[idx3(b, t, h, T, H)], local_db);
        }

        for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
          float term0 = shared.tmp_dk0[i];
          float term1 = 0.0f;
          float term2 = 0.0f;
          for (int j = 0; j < dv_local; ++j) {
            term1 += (alpha * shared.adj[i * dv_local + j]) * shared.tmp_dv0[j];
            term2 += shared.S0[i * dv_local + j] * shared.tmp_dv1[j];
          }
          atomicAdd(
              &grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)],
              beta_value * (term0 - term1 - term2));
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          int i = idx / dv_local;
          int j = idx - i * dv_local;
          shared.S1[idx] = alpha * shared.adj[idx] -
                           beta_value * shared.k[i] * shared.tmp_dv1[j];
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
          shared.adj[idx] = shared.S1[idx];
        }
        __syncthreads();
      }
    }
  }
  grid.sync();

  for (int64_t job = blockIdx.x; job < BT * H; job += gridDim.x) {
    int h = job % H;
    int64_t bt = job / H;
    int t = bt % T;
    int b = bt / T;

    float* q_preact_s = shmem;
    float* k_preact_s = shmem + Dk;
    float* reduce_s = shmem + 2 * Dk;
    float dot_q = 0.0f;
    float dot_k = 0.0f;
    float sq = 0.0f;
    float sk = 0.0f;
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      int q_channel = cq(h, i, Dk);
      int k_channel = ck(h, i, H, Dk);
      float q_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, q_channel, T, C, K)));
      float k_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, k_channel, T, C, K)));
      q_preact_s[i] = q_preact;
      k_preact_s[i] = k_preact;
      float q_value = silu(q_preact);
      float k_value = silu(k_preact);
      sq += q_value * q_value;
      sk += k_value * k_value;
    }
    block_sum2(sq, sk, reduce_s);

    float inv_q_norm = (sqrtf(sq) > EPS ? 1.0f / sqrtf(sq) : 1.0f / EPS);
    float inv_k_norm = (sqrtf(sk) > EPS ? 1.0f / sqrtf(sk) : 1.0f / EPS);
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      float dq = grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float dk = grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float q_value = silu(q_preact_s[i]) * inv_q_norm;
      float k_value = silu(k_preact_s[i]) * inv_k_norm;
      dot_q += dq * q_value;
      dot_k += dk * k_value;
    }
    block_sum2(dot_q, dot_k, reduce_s);
    for (int i = threadIdx.x; i < Dk; i += blockDim.x) {
      int q_channel = cq(h, i, Dk);
      int k_channel = ck(h, i, H, Dk);
      float dq = grad_q_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float dk = grad_k_norm_accum[idx4(b, t, h, i, T, H, Dk)];
      float q_preact = q_preact_s[i];
      float k_preact = k_preact_s[i];
      float q_value = silu(q_preact) * inv_q_norm;
      float k_value = silu(k_preact) * inv_k_norm;
      grad_pre[idx3(b, t, q_channel, T, C)] =
          f2b((dq - q_value * dot_q) * inv_q_norm * dsilu(q_preact));
      grad_pre[idx3(b, t, k_channel, T, C)] =
          f2b((dk - k_value * dot_k) * inv_k_norm * dsilu(k_preact));
    }
    for (int j = threadIdx.x; j < Dv; j += blockDim.x) {
      int v_channel = cv(h, j, H, Dk, Dv);
      float v_preact = b2f(f2b(conv_at(qkv, conv_w, b, t, v_channel, T, C, K)));
      float grad_value = b2f(grad_v_post[idx4(b, t, h, j, T, H, Dv)]);
      grad_pre[idx3(b, t, v_channel, T, C)] = f2b(grad_value * dsilu(v_preact));
    }
    __syncthreads();
  }
  grid.sync();

  for (int64_t idx = linear_tid; idx < BT * C; idx += linear_stride) {
    int c = idx % C;
    int64_t bt = idx / C;
    int tau = bt % T;
    int b = bt / T;
    float acc = 0.0f;
    for (int tap = 0; tap < K; ++tap) {
      int t = tau + (K - 1) - tap;
      if (t >= 0 && t < T) {
        acc += b2f(grad_pre[idx3(b, t, c, T, C)]) *
               b2f(conv_w[static_cast<int64_t>(c) * K + tap]);
      }
    }
    grad_qkv[idx] = f2b(acc);
  }
  for (int64_t idx = linear_tid; idx < static_cast<int64_t>(C) * K;
       idx += linear_stride) {
    int tap = idx % K;
    int c = idx / K;
    float acc = 0.0f;
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < T; ++t) {
        int tau = t - (K - 1) + tap;
        if (tau >= 0 && tau < T) {
          acc += b2f(grad_pre[idx3(b, t, c, T, C)]) *
                 b2f(qkv[idx3(b, tau, c, T, C)]);
        }
      }
    }
    grad_conv_w[idx] = f2b(acc);
  }
  grid.sync();

  for (int64_t h = linear_tid; h < H; h += linear_stride) {
    grad_A_log[h] = 0.0f;
    grad_dt_bias[h] = 0.0f;
  }
  grid.sync();

  float beta_scale = allow_neg_eigval ? 2.0f : 1.0f;
  for (int h = 0; h < H; ++h) {
    float local_grad_A = 0.0f;
    float local_grad_dt = 0.0f;
    float exp_A = expf(A_log[h]);
    for (int64_t bt = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         bt < BT;
         bt += linear_stride) {
      int64_t idx = bt * H + h;
      float z_value = b2f(g_pre[idx]) + dt_bias[h];
      float sig_z = sig(z_value);
      float grad_g_log_value = grad_g_log_accum[idx];
      float grad_dt_value = grad_g_log_value * (-exp_A * sig_z);
      grad_g_pre[idx] = f2b(grad_dt_value);
      float s = sig(b2f(beta_pre[idx]));
      grad_beta_pre[idx] =
          f2b(grad_beta_accum[idx] * beta_scale * s * (1.0f - s));
      local_grad_A += grad_g_log_value * (-exp_A * softplus(z_value));
      local_grad_dt += grad_dt_value;
    }
    block_sum2(local_grad_A, local_grad_dt, shmem);
    if (threadIdx.x == 0) {
      atomicAdd(&grad_A_log[h], local_grad_A);
      atomicAdd(&grad_dt_bias[h], local_grad_dt);
    }
    __syncthreads();
  }
}

size_t forward_shared_bytes(int Dk, int Dv) {
  size_t recurrence = forward_recurrence_bytes(Dk, Dv);
  size_t gemm =
      static_cast<size_t>(WARPS_PER_BLOCK * GEMM_TILE * GEMM_TILE) *
      sizeof(float);
  size_t reduction = static_cast<size_t>(2 * THREADS) * sizeof(float);
  size_t max_bytes = recurrence > gemm ? recurrence : gemm;
  return max_bytes > reduction ? max_bytes : reduction;
}

size_t backward_shared_bytes(int Dk, int Dv) {
  size_t recurrence = backward_recurrence_bytes(Dk, Dv);
  size_t gemm =
      static_cast<size_t>(WARPS_PER_BLOCK * GEMM_TILE * GEMM_TILE) *
      sizeof(float);
  size_t reduction = static_cast<size_t>(2 * THREADS) * sizeof(float);
  size_t max_bytes = recurrence > gemm ? recurrence : gemm;
  return max_bytes > reduction ? max_bytes : reduction;
}

template <typename KernelFn>
int cooperative_grid_blocks(KernelFn kernel, size_t shmem_bytes) {
  DeviceReport r = report();
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, THREADS, shmem_bytes));
  TORCH_CHECK(
      blocks_per_sm > 0,
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor returned zero active blocks");
  return blocks_per_sm * r.multiProcessorCount;
}

void validate_inputs(
    const torch::Tensor& x,
    const torch::Tensor& w_qkv,
    const torch::Tensor& w_a,
    const torch::Tensor& w_b,
    const torch::Tensor& w_g,
    const torch::Tensor& w_out,
    const torch::Tensor& conv_w,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K) {
  chk_bf16(x, "x");
  chk_bf16(w_qkv, "w_qkv");
  chk_bf16(w_a, "w_a");
  chk_bf16(w_b, "w_b");
  chk_bf16(w_g, "w_g");
  chk_bf16(w_out, "w_out");
  chk_bf16(conv_w, "conv_w");
  chk_f32(A_log, "A_log");
  chk_f32(dt_bias, "dt_bias");
  TORCH_CHECK(x.dim() == 3, "x must have shape (B, T, D)");
  int64_t D = x.size(2);
  int64_t C = H * (2 * Dk + Dv);
  int64_t P = H * Dv;
  TORCH_CHECK(
      D > 0 && D <= MAX_D && H > 0 && H <= MAX_H && Dk > 0 && Dk <= MAX_DK &&
          Dv > 0 && Dv <= MAX_DV && K > 0 && K <= MAX_K,
      "unsupported HGDN megakernel shape");
  TORCH_CHECK(
      w_qkv.sizes() == torch::IntArrayRef({C, D}),
      "w_qkv shape mismatch");
  TORCH_CHECK(
      w_a.sizes() == torch::IntArrayRef({H, D}), "w_a shape mismatch");
  TORCH_CHECK(
      w_b.sizes() == torch::IntArrayRef({H, D}), "w_b shape mismatch");
  TORCH_CHECK(
      w_g.sizes() == torch::IntArrayRef({P, D}), "w_g shape mismatch");
  TORCH_CHECK(
      w_out.sizes() == torch::IntArrayRef({D, P}), "w_out shape mismatch");
  TORCH_CHECK(
      conv_w.sizes() == torch::IntArrayRef({C, K}), "conv_w shape mismatch");
  TORCH_CHECK(A_log.numel() == H, "A_log shape mismatch");
  TORCH_CHECK(dt_bias.numel() == H, "dt_bias shape mismatch");
}

void validate_core_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& g_pre,
    const torch::Tensor& beta_pre,
    const torch::Tensor& g_out,
    const torch::Tensor& conv_w,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K) {
  chk_bf16(qkv, "qkv");
  chk_bf16(g_pre, "g_pre");
  chk_bf16(beta_pre, "beta_pre");
  chk_bf16(g_out, "g_out");
  chk_bf16(conv_w, "conv_w");
  chk_f32(A_log, "A_log");
  chk_f32(dt_bias, "dt_bias");
  TORCH_CHECK(qkv.dim() == 3, "qkv must have shape (B, T, C)");
  TORCH_CHECK(g_pre.dim() == 3, "g_pre must have shape (B, T, H)");
  TORCH_CHECK(beta_pre.dim() == 3, "beta_pre must have shape (B, T, H)");
  TORCH_CHECK(g_out.dim() == 4, "g_out must have shape (B, T, H, Dv)");
  int64_t B = qkv.size(0);
  int64_t T = qkv.size(1);
  int64_t C = H * (2 * Dk + Dv);
  TORCH_CHECK(
      H > 0 && H <= MAX_H && Dk > 0 && Dk <= MAX_DK && Dv > 0 &&
          Dv <= MAX_DV && K > 0 && K <= MAX_K,
      "unsupported HGDN core kernel shape");
  TORCH_CHECK(
      qkv.sizes() == torch::IntArrayRef({B, T, C}), "qkv shape mismatch");
  TORCH_CHECK(
      g_pre.sizes() == torch::IntArrayRef({B, T, H}), "g_pre shape mismatch");
  TORCH_CHECK(
      beta_pre.sizes() == torch::IntArrayRef({B, T, H}),
      "beta_pre shape mismatch");
  TORCH_CHECK(
      g_out.sizes() == torch::IntArrayRef({B, T, H, Dv}),
      "g_out shape mismatch");
  TORCH_CHECK(
      conv_w.sizes() == torch::IntArrayRef({C, K}), "conv_w shape mismatch");
  TORCH_CHECK(A_log.numel() == H, "A_log shape mismatch");
  TORCH_CHECK(dt_bias.numel() == H, "dt_bias shape mismatch");
}

std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor w_a,
    torch::Tensor w_b,
    torch::Tensor w_g,
    torch::Tensor w_out,
    torch::Tensor conv_w,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K,
    int64_t rec_chunk_t,
    bool allow_neg_eigval) {
  c10::cuda::CUDAGuard guard(x.device());
  validate_runtime_device();
  validate_inputs(
      x, w_qkv, w_a, w_b, w_g, w_out, conv_w, A_log, dt_bias, H, Dk, Dv, K);
  TORCH_CHECK(
      rec_chunk_t > 0 && rec_chunk_t <= REC_CHUNK_T_MAX,
      "HGDN megakernel rec_chunk_t must be in [1, ",
      REC_CHUNK_T_MAX,
      "], got ",
      rec_chunk_t);

  int64_t B = x.size(0);
  int64_t T = x.size(1);
  int64_t D = x.size(2);
  int64_t P = H * Dv;
  int64_t C = H * (2 * Dk + Dv);
  int64_t NChunks = (T + rec_chunk_t - 1) / rec_chunk_t;
  auto bf16_options = x.options();
  auto f32_options = x.options().dtype(torch::kFloat32);

  auto y = torch::empty({B, T, D}, bf16_options);
  auto qkv = torch::empty({B, T, C}, bf16_options);
  auto pre_tmp = torch::empty({B, T, C}, bf16_options);
  auto g_pre = torch::empty({B, T, H}, bf16_options);
  auto beta_pre = torch::empty({B, T, H}, bf16_options);
  auto g_log = torch::empty({B, T, H}, bf16_options);
  auto beta = torch::empty({B, T, H}, bf16_options);
  auto g_out = torch::empty({B, T, P}, bf16_options);
  auto o_raw = torch::empty({B, T, H, Dv}, bf16_options);
  auto z_tmp = torch::empty({B, T, P}, bf16_options);
  auto state_ckpt = torch::empty({B, NChunks, H, Dk, Dv}, f32_options);

  int iB = static_cast<int>(B);
  int iT = static_cast<int>(T);
  int iD = static_cast<int>(D);
  int iH = static_cast<int>(H);
  int iDk = static_cast<int>(Dk);
  int iDv = static_cast<int>(Dv);
  int iK = static_cast<int>(K);
  int iNChunks = static_cast<int>(NChunks);
  int iChunkT = static_cast<int>(rec_chunk_t);
  int iAllow = allow_neg_eigval ? 1 : 0;
  size_t shmem_bytes = forward_shared_bytes(iDk, iDv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      hgdn_forward_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes)));
  int blocks = cooperative_grid_blocks(hgdn_forward_bf16_kernel, shmem_bytes);

  const bf16* x_ptr = cbptr(x);
  const bf16* w_qkv_ptr = cbptr(w_qkv);
  const bf16* w_a_ptr = cbptr(w_a);
  const bf16* w_b_ptr = cbptr(w_b);
  const bf16* w_g_ptr = cbptr(w_g);
  const bf16* w_out_ptr = cbptr(w_out);
  const bf16* conv_w_ptr = cbptr(conv_w);
  const float* A_log_ptr = A_log.data_ptr<float>();
  const float* dt_bias_ptr = dt_bias.data_ptr<float>();
  bf16* y_ptr = bptr(y);
  bf16* qkv_ptr = bptr(qkv);
  bf16* pre_tmp_ptr = bptr(pre_tmp);
  bf16* g_pre_ptr = bptr(g_pre);
  bf16* beta_pre_ptr = bptr(beta_pre);
  bf16* g_log_ptr = bptr(g_log);
  bf16* beta_ptr = bptr(beta);
  bf16* g_out_ptr = bptr(g_out);
  bf16* o_raw_ptr = bptr(o_raw);
  bf16* z_tmp_ptr = bptr(z_tmp);
  float* state_ckpt_ptr = state_ckpt.data_ptr<float>();

  void* args[] = {
      &x_ptr,
      &w_qkv_ptr,
      &w_a_ptr,
      &w_b_ptr,
      &w_g_ptr,
      &w_out_ptr,
      &conv_w_ptr,
      &A_log_ptr,
      &dt_bias_ptr,
      &y_ptr,
      &qkv_ptr,
      &pre_tmp_ptr,
      &g_pre_ptr,
      &beta_pre_ptr,
      &g_log_ptr,
      &beta_ptr,
      &g_out_ptr,
      &o_raw_ptr,
      &z_tmp_ptr,
      &state_ckpt_ptr,
      &iB,
      &iT,
      &iD,
      &iH,
      &iDk,
      &iDv,
      &iK,
      &iNChunks,
      &iChunkT,
      &iAllow,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(hgdn_forward_bf16_kernel),
      dim3(blocks),
      dim3(THREADS),
      args,
      shmem_bytes,
      at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      y,      qkv,      g_pre,  beta_pre, g_log,
      beta,   g_out,    o_raw,  state_ckpt,
  };
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_y,
    torch::Tensor x,
    torch::Tensor w_qkv,
    torch::Tensor w_a,
    torch::Tensor w_b,
    torch::Tensor w_g,
    torch::Tensor w_out,
    torch::Tensor conv_w,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor qkv,
    torch::Tensor g_pre,
    torch::Tensor beta_pre,
    torch::Tensor g_log,
    torch::Tensor beta,
    torch::Tensor g_out,
    torch::Tensor o_raw,
    torch::Tensor state_ckpt,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K,
    int64_t rec_chunk_t,
    bool allow_neg_eigval) {
  c10::cuda::CUDAGuard guard(x.device());
  validate_runtime_device();
  validate_inputs(
      x, w_qkv, w_a, w_b, w_g, w_out, conv_w, A_log, dt_bias, H, Dk, Dv, K);
  chk_bf16(grad_y, "grad_y");
  TORCH_CHECK(
      rec_chunk_t > 0 && rec_chunk_t <= REC_CHUNK_T_MAX,
      "HGDN megakernel rec_chunk_t must be in [1, ",
      REC_CHUNK_T_MAX,
      "], got ",
      rec_chunk_t);

  int64_t B = x.size(0);
  int64_t T = x.size(1);
  int64_t D = x.size(2);
  int64_t P = H * Dv;
  int64_t C = H * (2 * Dk + Dv);
  int64_t NChunks = (T + rec_chunk_t - 1) / rec_chunk_t;
  auto bf16_options = x.options();
  auto f32_options = x.options().dtype(torch::kFloat32);

  auto grad_x = torch::empty_like(x);
  auto grad_x_accum = torch::empty({B, T, D}, f32_options);
  auto grad_w_qkv = torch::empty_like(w_qkv);
  auto grad_w_a = torch::empty_like(w_a);
  auto grad_w_b = torch::empty_like(w_b);
  auto grad_w_g = torch::empty_like(w_g);
  auto grad_w_out = torch::empty_like(w_out);
  auto grad_conv_w = torch::empty_like(conv_w);
  auto grad_A_log = torch::empty_like(A_log);
  auto grad_dt_bias = torch::empty_like(dt_bias);

  auto grad_z = torch::empty({B, T, P}, bf16_options);
  auto grad_o_raw = torch::empty({B, T, H, Dv}, bf16_options);
  auto grad_g_out = torch::empty({B, T, P}, bf16_options);
  auto grad_q_norm_accum = torch::empty({B, T, H, Dk}, f32_options);
  auto grad_k_norm_accum = torch::empty({B, T, H, Dk}, f32_options);
  auto grad_g_log_accum = torch::empty({B, T, H}, f32_options);
  auto grad_beta_accum = torch::empty({B, T, H}, f32_options);
  auto grad_v_post = torch::empty({B, T, H, Dv}, bf16_options);
  auto grad_pre = torch::empty({B, T, C}, bf16_options);
  auto grad_qkv = torch::empty({B, T, C}, bf16_options);
  auto grad_g_pre = torch::empty({B, T, H}, bf16_options);
  auto grad_beta_pre = torch::empty({B, T, H}, bf16_options);

  int iB = static_cast<int>(B);
  int iT = static_cast<int>(T);
  int iD = static_cast<int>(D);
  int iH = static_cast<int>(H);
  int iDk = static_cast<int>(Dk);
  int iDv = static_cast<int>(Dv);
  int iK = static_cast<int>(K);
  int iNChunks = static_cast<int>(NChunks);
  int iChunkT = static_cast<int>(rec_chunk_t);
  int iAllow = allow_neg_eigval ? 1 : 0;
  size_t shmem_bytes = backward_shared_bytes(iDk, iDv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      hgdn_backward_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes)));
  int blocks = cooperative_grid_blocks(hgdn_backward_bf16_kernel, shmem_bytes);

  const bf16* grad_y_ptr = cbptr(grad_y);
  const bf16* x_ptr = cbptr(x);
  const bf16* w_qkv_ptr = cbptr(w_qkv);
  const bf16* w_a_ptr = cbptr(w_a);
  const bf16* w_b_ptr = cbptr(w_b);
  const bf16* w_g_ptr = cbptr(w_g);
  const bf16* w_out_ptr = cbptr(w_out);
  const bf16* conv_w_ptr = cbptr(conv_w);
  const float* A_log_ptr = A_log.data_ptr<float>();
  const float* dt_bias_ptr = dt_bias.data_ptr<float>();
  const bf16* qkv_ptr = cbptr(qkv);
  const bf16* g_pre_ptr = cbptr(g_pre);
  const bf16* beta_pre_ptr = cbptr(beta_pre);
  const bf16* g_log_ptr = cbptr(g_log);
  const bf16* beta_ptr = cbptr(beta);
  const bf16* g_out_ptr = cbptr(g_out);
  const bf16* o_raw_ptr = cbptr(o_raw);
  const float* state_ckpt_ptr = state_ckpt.data_ptr<float>();

  float* grad_q_norm_accum_ptr = grad_q_norm_accum.data_ptr<float>();
  float* grad_k_norm_accum_ptr = grad_k_norm_accum.data_ptr<float>();
  float* grad_g_log_accum_ptr = grad_g_log_accum.data_ptr<float>();
  float* grad_beta_accum_ptr = grad_beta_accum.data_ptr<float>();
  float* grad_x_accum_ptr = grad_x_accum.data_ptr<float>();
  bf16* grad_x_ptr = bptr(grad_x);
  bf16* grad_w_qkv_ptr = bptr(grad_w_qkv);
  bf16* grad_w_a_ptr = bptr(grad_w_a);
  bf16* grad_w_b_ptr = bptr(grad_w_b);
  bf16* grad_w_g_ptr = bptr(grad_w_g);
  bf16* grad_w_out_ptr = bptr(grad_w_out);
  bf16* grad_conv_w_ptr = bptr(grad_conv_w);
  float* grad_A_log_ptr = grad_A_log.data_ptr<float>();
  float* grad_dt_bias_ptr = grad_dt_bias.data_ptr<float>();
  bf16* grad_z_ptr = bptr(grad_z);
  bf16* grad_o_raw_ptr = bptr(grad_o_raw);
  bf16* grad_g_out_ptr = bptr(grad_g_out);
  bf16* grad_v_post_ptr = bptr(grad_v_post);
  bf16* grad_pre_ptr = bptr(grad_pre);
  bf16* grad_qkv_ptr = bptr(grad_qkv);
  bf16* grad_g_pre_ptr = bptr(grad_g_pre);
  bf16* grad_beta_pre_ptr = bptr(grad_beta_pre);

  void* args[] = {
      &grad_y_ptr,       &x_ptr,           &w_qkv_ptr,        &w_a_ptr,
      &w_b_ptr,          &w_g_ptr,         &w_out_ptr,        &conv_w_ptr,
      &A_log_ptr,        &dt_bias_ptr,     &qkv_ptr,          &g_pre_ptr,
      &beta_pre_ptr,     &g_log_ptr,       &beta_ptr,
      &g_out_ptr,        &o_raw_ptr,       &state_ckpt_ptr,
      &grad_q_norm_accum_ptr,
      &grad_k_norm_accum_ptr,              &grad_g_log_accum_ptr,
      &grad_beta_accum_ptr,                &grad_x_accum_ptr,
      &grad_x_ptr,       &grad_w_qkv_ptr,  &grad_w_a_ptr,       &grad_w_b_ptr,
      &grad_w_g_ptr,     &grad_w_out_ptr,
      &grad_conv_w_ptr,  &grad_A_log_ptr,  &grad_dt_bias_ptr, &grad_z_ptr,
      &grad_o_raw_ptr,   &grad_g_out_ptr,  &grad_v_post_ptr,  &grad_pre_ptr,
      &grad_qkv_ptr,     &grad_g_pre_ptr,  &grad_beta_pre_ptr,
      &iB,               &iT,              &iD,               &iH,
      &iDk,              &iDv,             &iK,               &iNChunks,       &iChunkT,
      &iAllow,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(hgdn_backward_bf16_kernel),
      dim3(blocks),
      dim3(THREADS),
      args,
      shmem_bytes,
      at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      grad_x,
      grad_w_qkv,
      grad_w_a,
      grad_w_b,
      grad_w_g,
      grad_w_out,
      grad_conv_w,
      grad_A_log,
      grad_dt_bias,
  };
}

std::vector<torch::Tensor> core_forward(
    torch::Tensor qkv,
    torch::Tensor g_pre,
    torch::Tensor beta_pre,
    torch::Tensor g_out,
    torch::Tensor conv_w,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K,
    int64_t rec_chunk_t,
    bool allow_neg_eigval) {
  c10::cuda::CUDAGuard guard(qkv.device());
  validate_runtime_device();
  validate_core_inputs(qkv, g_pre, beta_pre, g_out, conv_w, A_log, dt_bias, H, Dk, Dv, K);
  TORCH_CHECK(
      rec_chunk_t > 0 && rec_chunk_t <= REC_CHUNK_T_MAX,
      "HGDN core kernel rec_chunk_t must be in [1, ",
      REC_CHUNK_T_MAX,
      "], got ",
      rec_chunk_t);

  int64_t B = qkv.size(0);
  int64_t T = qkv.size(1);
  int64_t NChunks = (T + rec_chunk_t - 1) / rec_chunk_t;
  auto bf16_options = qkv.options();
  auto f32_options = qkv.options().dtype(torch::kFloat32);

  auto z = torch::empty({B, T, H, Dv}, bf16_options);
  auto g_log = torch::empty({B, T, H}, bf16_options);
  auto beta = torch::empty({B, T, H}, bf16_options);
  auto o_raw = torch::empty({B, T, H, Dv}, bf16_options);
  auto state_ckpt = torch::empty({B, NChunks, H, Dk, Dv}, f32_options);

  int iB = static_cast<int>(B);
  int iT = static_cast<int>(T);
  int iH = static_cast<int>(H);
  int iDk = static_cast<int>(Dk);
  int iDv = static_cast<int>(Dv);
  int iK = static_cast<int>(K);
  int iNChunks = static_cast<int>(NChunks);
  int iChunkT = static_cast<int>(rec_chunk_t);
  int iAllow = allow_neg_eigval ? 1 : 0;
  size_t shmem_bytes = forward_shared_bytes(iDk, iDv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      hgdn_core_forward_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes)));
  int blocks = cooperative_grid_blocks(hgdn_core_forward_bf16_kernel, shmem_bytes);

  const bf16* qkv_ptr = cbptr(qkv);
  const bf16* g_pre_ptr = cbptr(g_pre);
  const bf16* beta_pre_ptr = cbptr(beta_pre);
  const bf16* g_out_ptr = cbptr(g_out);
  const bf16* conv_w_ptr = cbptr(conv_w);
  const float* A_log_ptr = A_log.data_ptr<float>();
  const float* dt_bias_ptr = dt_bias.data_ptr<float>();
  bf16* z_ptr = bptr(z);
  bf16* g_log_ptr = bptr(g_log);
  bf16* beta_ptr = bptr(beta);
  bf16* o_raw_ptr = bptr(o_raw);
  float* state_ckpt_ptr = state_ckpt.data_ptr<float>();

  void* args[] = {
      &qkv_ptr,
      &g_pre_ptr,
      &beta_pre_ptr,
      &g_out_ptr,
      &conv_w_ptr,
      &A_log_ptr,
      &dt_bias_ptr,
      &z_ptr,
      &g_log_ptr,
      &beta_ptr,
      &o_raw_ptr,
      &state_ckpt_ptr,
      &iB,
      &iT,
      &iH,
      &iDk,
      &iDv,
      &iK,
      &iNChunks,
      &iChunkT,
      &iAllow,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(hgdn_core_forward_bf16_kernel),
      dim3(blocks),
      dim3(THREADS),
      args,
      shmem_bytes,
      at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {z, g_log, beta, o_raw, state_ckpt};
}

std::vector<torch::Tensor> core_backward(
    torch::Tensor grad_z,
    torch::Tensor qkv,
    torch::Tensor g_pre,
    torch::Tensor beta_pre,
    torch::Tensor g_out,
    torch::Tensor conv_w,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    torch::Tensor g_log,
    torch::Tensor beta,
    torch::Tensor o_raw,
    torch::Tensor state_ckpt,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t K,
    int64_t rec_chunk_t,
    bool allow_neg_eigval) {
  c10::cuda::CUDAGuard guard(qkv.device());
  validate_runtime_device();
  validate_core_inputs(qkv, g_pre, beta_pre, g_out, conv_w, A_log, dt_bias, H, Dk, Dv, K);
  chk_bf16(grad_z, "grad_z");
  TORCH_CHECK(
      grad_z.sizes() == torch::IntArrayRef({qkv.size(0), qkv.size(1), H, Dv}),
      "grad_z shape mismatch");
  TORCH_CHECK(
      rec_chunk_t > 0 && rec_chunk_t <= REC_CHUNK_T_MAX,
      "HGDN core kernel rec_chunk_t must be in [1, ",
      REC_CHUNK_T_MAX,
      "], got ",
      rec_chunk_t);

  int64_t B = qkv.size(0);
  int64_t T = qkv.size(1);
  int64_t C = H * (2 * Dk + Dv);
  int64_t NChunks = (T + rec_chunk_t - 1) / rec_chunk_t;
  auto bf16_options = qkv.options();
  auto f32_options = qkv.options().dtype(torch::kFloat32);

  auto grad_qkv = torch::empty({B, T, C}, bf16_options);
  auto grad_g_pre = torch::empty_like(g_pre);
  auto grad_beta_pre = torch::empty_like(beta_pre);
  auto grad_g_out = torch::empty_like(g_out);
  auto grad_conv_w = torch::empty_like(conv_w);
  auto grad_A_log = torch::empty_like(A_log);
  auto grad_dt_bias = torch::empty_like(dt_bias);

  auto grad_o_raw = torch::empty({B, T, H, Dv}, bf16_options);
  auto grad_q_norm_accum = torch::empty({B, T, H, Dk}, f32_options);
  auto grad_k_norm_accum = torch::empty({B, T, H, Dk}, f32_options);
  auto grad_g_log_accum = torch::empty({B, T, H}, f32_options);
  auto grad_beta_accum = torch::empty({B, T, H}, f32_options);
  auto grad_v_post = torch::empty({B, T, H, Dv}, bf16_options);
  auto grad_pre = torch::empty({B, T, C}, bf16_options);

  int iB = static_cast<int>(B);
  int iT = static_cast<int>(T);
  int iH = static_cast<int>(H);
  int iDk = static_cast<int>(Dk);
  int iDv = static_cast<int>(Dv);
  int iK = static_cast<int>(K);
  int iNChunks = static_cast<int>(NChunks);
  int iChunkT = static_cast<int>(rec_chunk_t);
  int iAllow = allow_neg_eigval ? 1 : 0;
  size_t shmem_bytes = backward_shared_bytes(iDk, iDv);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      hgdn_core_backward_bf16_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shmem_bytes)));
  int blocks = cooperative_grid_blocks(hgdn_core_backward_bf16_kernel, shmem_bytes);

  const bf16* grad_z_ptr = cbptr(grad_z);
  const bf16* qkv_ptr = cbptr(qkv);
  const bf16* g_pre_ptr = cbptr(g_pre);
  const bf16* beta_pre_ptr = cbptr(beta_pre);
  const bf16* g_out_ptr = cbptr(g_out);
  const bf16* conv_w_ptr = cbptr(conv_w);
  const float* A_log_ptr = A_log.data_ptr<float>();
  const float* dt_bias_ptr = dt_bias.data_ptr<float>();
  const bf16* g_log_ptr = cbptr(g_log);
  const bf16* beta_ptr = cbptr(beta);
  const bf16* o_raw_ptr = cbptr(o_raw);
  const float* state_ckpt_ptr = state_ckpt.data_ptr<float>();

  float* grad_q_norm_accum_ptr = grad_q_norm_accum.data_ptr<float>();
  float* grad_k_norm_accum_ptr = grad_k_norm_accum.data_ptr<float>();
  float* grad_g_log_accum_ptr = grad_g_log_accum.data_ptr<float>();
  float* grad_beta_accum_ptr = grad_beta_accum.data_ptr<float>();
  bf16* grad_g_out_ptr = bptr(grad_g_out);
  bf16* grad_o_raw_ptr = bptr(grad_o_raw);
  bf16* grad_v_post_ptr = bptr(grad_v_post);
  bf16* grad_pre_ptr = bptr(grad_pre);
  bf16* grad_qkv_ptr = bptr(grad_qkv);
  bf16* grad_g_pre_ptr = bptr(grad_g_pre);
  bf16* grad_beta_pre_ptr = bptr(grad_beta_pre);
  bf16* grad_conv_w_ptr = bptr(grad_conv_w);
  float* grad_A_log_ptr = grad_A_log.data_ptr<float>();
  float* grad_dt_bias_ptr = grad_dt_bias.data_ptr<float>();

  void* args[] = {
      &grad_z_ptr,
      &qkv_ptr,
      &g_pre_ptr,
      &beta_pre_ptr,
      &g_out_ptr,
      &conv_w_ptr,
      &A_log_ptr,
      &dt_bias_ptr,
      &g_log_ptr,
      &beta_ptr,
      &o_raw_ptr,
      &state_ckpt_ptr,
      &grad_q_norm_accum_ptr,
      &grad_k_norm_accum_ptr,
      &grad_g_log_accum_ptr,
      &grad_beta_accum_ptr,
      &grad_g_out_ptr,
      &grad_o_raw_ptr,
      &grad_v_post_ptr,
      &grad_pre_ptr,
      &grad_qkv_ptr,
      &grad_g_pre_ptr,
      &grad_beta_pre_ptr,
      &grad_conv_w_ptr,
      &grad_A_log_ptr,
      &grad_dt_bias_ptr,
      &iB,
      &iT,
      &iH,
      &iDk,
      &iDv,
      &iK,
      &iNChunks,
      &iChunkT,
      &iAllow,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(hgdn_core_backward_bf16_kernel),
      dim3(blocks),
      dim3(THREADS),
      args,
      shmem_bytes,
      at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      grad_qkv,
      grad_g_pre,
      grad_beta_pre,
      grad_g_out,
      grad_conv_w,
      grad_A_log,
      grad_dt_bias,
  };
}

int64_t rec_chunk_t_max() {
  return REC_CHUNK_T_MAX;
}

std::string build_config_json() {
  std::ostringstream out;
  out << "{"
      << "\"threads\":" << THREADS << ","
      << "\"rec_v_tile\":" << REC_V_TILE << ","
      << "\"rec_chunk_t_max\":" << REC_CHUNK_T_MAX << ","
      << "\"gemm_atb_split_m_threshold\":"
      << GEMM_ATB_BLOCK_SPLIT_M_THRESHOLD << "}";
  return out.str();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_report", &device_report_string);
  m.def("rec_chunk_t_max", &rec_chunk_t_max);
  m.def("build_config_json", &build_config_json);
  m.def("forward", &forward);
  m.def("backward", &backward);
  m.def("core_forward", &core_forward);
  m.def("core_backward", &core_backward);
}
