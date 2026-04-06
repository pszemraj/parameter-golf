#include <torch/extension.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float x) {
  return static_cast<scalar_t>(x);
}

template <typename scalar_t>
__device__ __forceinline__ float silu_from_preact(scalar_t x) {
  const float xf = to_float(x);
  const float sig = 1.0f / (1.0f + expf(-xf));
  return xf * sig;
}

template <typename scalar_t>
__device__ __forceinline__ float silu_backward_from_preact(scalar_t x) {
  const float xf = to_float(x);
  const float sig = 1.0f / (1.0f + expf(-xf));
  return sig * (1.0f + xf * (1.0f - sig));
}

template <typename scalar_t>
__global__ void causal_dwconv_preact_forward_kernel(
    const scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ preact,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t K) {
  const int64_t total = B * T * C;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = idx % C;
    const int64_t tmp = idx / C;
    const int64_t t = tmp % T;
    const int64_t b = tmp / T;
    float acc = 0.0f;
    const int64_t start_t = t - (K - 1);
    for (int64_t k = 0; k < K; ++k) {
      const int64_t tau = start_t + k;
      if (tau >= 0 && tau < T) {
        const int64_t x_idx = ((b * T + tau) * C + c);
        acc += to_float(qkv[x_idx]) * to_float(weight[c * K + k]);
      }
    }
    preact[idx] = from_float<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void silu_from_preact_kernel(
    const scalar_t* __restrict__ preact,
    scalar_t* __restrict__ packed_out,
    int64_t total) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    packed_out[idx] = from_float<scalar_t>(silu_from_preact(preact[idx]));
  }
}

template <typename scalar_t>
__global__ void silu_grad_from_preact_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ preact,
    scalar_t* __restrict__ grad_preact,
    int64_t total) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    grad_preact[idx] = from_float<scalar_t>(
        to_float(grad_out[idx]) * silu_backward_from_preact(preact[idx]));
  }
}

template <typename scalar_t>
__global__ void split_norm_from_preact_kernel(
    const scalar_t* __restrict__ preact,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    float* __restrict__ inv_q,
    float* __restrict__ inv_k,
    int64_t BT,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t C,
    float eps) {
  const int64_t head_idx = blockIdx.x;
  const int64_t bt = head_idx / H;
  const int64_t h = head_idx % H;
  if (bt >= BT) {
    return;
  }
  const int64_t q_dim = H * Dk;
  const int64_t k_dim = H * Dk;
  const int64_t base = bt * C;
  const int64_t q_base = base + h * Dk;
  const int64_t k_base = base + q_dim + h * Dk;
  const int64_t v_base = base + q_dim + k_dim + h * Dv;
  const int64_t out_q_base = head_idx * Dk;
  const int64_t out_k_base = head_idx * Dk;
  const int64_t out_v_base = head_idx * Dv;

  extern __shared__ float shmem[];
  float* sh_q = shmem;
  float* sh_k = shmem + blockDim.x;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    const float qv = silu_from_preact(preact[q_base + d]);
    const float kv = silu_from_preact(preact[k_base + d]);
    q_sum += qv * qv;
    k_sum += kv * kv;
  }
  sh_q[threadIdx.x] = q_sum;
  sh_k[threadIdx.x] = k_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_q[threadIdx.x] += sh_q[threadIdx.x + stride];
      sh_k[threadIdx.x] += sh_k[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float q_inv = rsqrtf(sh_q[0] + eps);
  const float k_inv = rsqrtf(sh_k[0] + eps);
  if (threadIdx.x == 0) {
    inv_q[head_idx] = q_inv;
    inv_k[head_idx] = k_inv;
  }

  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    const float qv = silu_from_preact(preact[q_base + d]) * q_inv;
    const float kv = silu_from_preact(preact[k_base + d]) * k_inv;
    q_out[out_q_base + d] = from_float<scalar_t>(qv);
    k_out[out_k_base + d] = from_float<scalar_t>(kv);
  }
  for (int64_t d = threadIdx.x; d < Dv; d += blockDim.x) {
    const float vv = silu_from_preact(preact[v_base + d]);
    v_out[out_v_base + d] = from_float<scalar_t>(vv);
  }
}

template <typename scalar_t>
__global__ void build_grad_preact_kernel(
    const scalar_t* __restrict__ grad_q,
    const scalar_t* __restrict__ grad_k,
    const scalar_t* __restrict__ grad_v,
    const scalar_t* __restrict__ preact,
    const scalar_t* __restrict__ q_norm,
    const scalar_t* __restrict__ k_norm,
    const float* __restrict__ inv_q,
    const float* __restrict__ inv_k,
    scalar_t* __restrict__ grad_preact,
    int64_t BT,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t C) {
  const int64_t head_idx = blockIdx.x;
  const int64_t bt = head_idx / H;
  const int64_t h = head_idx % H;
  if (bt >= BT) {
    return;
  }
  const int64_t q_dim = H * Dk;
  const int64_t k_dim = H * Dk;
  const int64_t base = bt * C;
  const int64_t q_base = base + h * Dk;
  const int64_t k_base = base + q_dim + h * Dk;
  const int64_t v_base = base + q_dim + k_dim + h * Dv;
  const int64_t out_q_base = head_idx * Dk;
  const int64_t out_k_base = head_idx * Dk;
  const int64_t out_v_base = head_idx * Dv;

  extern __shared__ float shmem[];
  float* sh_q = shmem;
  float* sh_k = shmem + blockDim.x;

  float dot_q = 0.0f;
  float dot_k = 0.0f;
  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    dot_q += to_float(grad_q[out_q_base + d]) * to_float(q_norm[out_q_base + d]);
    dot_k += to_float(grad_k[out_k_base + d]) * to_float(k_norm[out_k_base + d]);
  }
  sh_q[threadIdx.x] = dot_q;
  sh_k[threadIdx.x] = dot_k;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_q[threadIdx.x] += sh_q[threadIdx.x + stride];
      sh_k[threadIdx.x] += sh_k[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float q_inv = inv_q[head_idx];
  const float k_inv = inv_k[head_idx];
  const float q_dot = sh_q[0];
  const float k_dot = sh_k[0];

  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    const float qn = to_float(q_norm[out_q_base + d]);
    const float kn = to_float(k_norm[out_k_base + d]);
    const float gq = to_float(grad_q[out_q_base + d]);
    const float gk = to_float(grad_k[out_k_base + d]);
    const float q_pre = (gq - qn * q_dot) * q_inv;
    const float k_pre = (gk - kn * k_dot) * k_inv;
    grad_preact[q_base + d] = from_float<scalar_t>(
        q_pre * silu_backward_from_preact(preact[q_base + d]));
    grad_preact[k_base + d] = from_float<scalar_t>(
        k_pre * silu_backward_from_preact(preact[k_base + d]));
  }
  for (int64_t d = threadIdx.x; d < Dv; d += blockDim.x) {
    const float gv = to_float(grad_v[out_v_base + d]);
    grad_preact[v_base + d] = from_float<scalar_t>(
        gv * silu_backward_from_preact(preact[v_base + d]));
  }
}

template <typename scalar_t>
__global__ void split_l2norm_from_packed_kernel(
    const scalar_t* __restrict__ packed,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    float* __restrict__ inv_q,
    float* __restrict__ inv_k,
    int64_t BT,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t C,
    float eps) {
  const int64_t head_idx = blockIdx.x;
  const int64_t bt = head_idx / H;
  const int64_t h = head_idx % H;
  if (bt >= BT) {
    return;
  }
  const int64_t q_dim = H * Dk;
  const int64_t k_dim = H * Dk;
  const int64_t base = bt * C;
  const int64_t q_base = base + h * Dk;
  const int64_t k_base = base + q_dim + h * Dk;
  const int64_t v_base = base + q_dim + k_dim + h * Dv;
  const int64_t out_q_base = head_idx * Dk;
  const int64_t out_k_base = head_idx * Dk;
  const int64_t out_v_base = head_idx * Dv;

  extern __shared__ float shmem[];
  float* sh_q = shmem;
  float* sh_k = shmem + blockDim.x;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    const float qv = to_float(packed[q_base + d]);
    const float kv = to_float(packed[k_base + d]);
    q_sum += qv * qv;
    k_sum += kv * kv;
  }
  sh_q[threadIdx.x] = q_sum;
  sh_k[threadIdx.x] = k_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_q[threadIdx.x] += sh_q[threadIdx.x + stride];
      sh_k[threadIdx.x] += sh_k[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float q_inv = rsqrtf(sh_q[0] + eps);
  const float k_inv = rsqrtf(sh_k[0] + eps);
  if (threadIdx.x == 0) {
    inv_q[head_idx] = q_inv;
    inv_k[head_idx] = k_inv;
  }

  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    q_out[out_q_base + d] = from_float<scalar_t>(to_float(packed[q_base + d]) * q_inv);
    k_out[out_k_base + d] = from_float<scalar_t>(to_float(packed[k_base + d]) * k_inv);
  }
  for (int64_t d = threadIdx.x; d < Dv; d += blockDim.x) {
    v_out[out_v_base + d] = packed[v_base + d];
  }
}

template <typename scalar_t>
__global__ void build_grad_packed_from_norm_kernel(
    const scalar_t* __restrict__ grad_q,
    const scalar_t* __restrict__ grad_k,
    const scalar_t* __restrict__ grad_v,
    const scalar_t* __restrict__ q_norm,
    const scalar_t* __restrict__ k_norm,
    const float* __restrict__ inv_q,
    const float* __restrict__ inv_k,
    scalar_t* __restrict__ grad_packed,
    int64_t BT,
    int64_t H,
    int64_t Dk,
    int64_t Dv,
    int64_t C) {
  const int64_t head_idx = blockIdx.x;
  const int64_t bt = head_idx / H;
  const int64_t h = head_idx % H;
  if (bt >= BT) {
    return;
  }
  const int64_t q_dim = H * Dk;
  const int64_t k_dim = H * Dk;
  const int64_t base = bt * C;
  const int64_t q_base = base + h * Dk;
  const int64_t k_base = base + q_dim + h * Dk;
  const int64_t v_base = base + q_dim + k_dim + h * Dv;
  const int64_t out_q_base = head_idx * Dk;
  const int64_t out_k_base = head_idx * Dk;
  const int64_t out_v_base = head_idx * Dv;

  extern __shared__ float shmem[];
  float* sh_q = shmem;
  float* sh_k = shmem + blockDim.x;

  float dot_q = 0.0f;
  float dot_k = 0.0f;
  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    dot_q += to_float(grad_q[out_q_base + d]) * to_float(q_norm[out_q_base + d]);
    dot_k += to_float(grad_k[out_k_base + d]) * to_float(k_norm[out_k_base + d]);
  }
  sh_q[threadIdx.x] = dot_q;
  sh_k[threadIdx.x] = dot_k;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_q[threadIdx.x] += sh_q[threadIdx.x + stride];
      sh_k[threadIdx.x] += sh_k[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float q_inv = inv_q[head_idx];
  const float k_inv = inv_k[head_idx];
  const float q_dot = sh_q[0];
  const float k_dot = sh_k[0];

  for (int64_t d = threadIdx.x; d < Dk; d += blockDim.x) {
    const float qn = to_float(q_norm[out_q_base + d]);
    const float kn = to_float(k_norm[out_k_base + d]);
    const float gq = to_float(grad_q[out_q_base + d]);
    const float gk = to_float(grad_k[out_k_base + d]);
    grad_packed[q_base + d] = from_float<scalar_t>((gq - qn * q_dot) * q_inv);
    grad_packed[k_base + d] = from_float<scalar_t>((gk - kn * k_dot) * k_inv);
  }
  for (int64_t d = threadIdx.x; d < Dv; d += blockDim.x) {
    grad_packed[v_base + d] = grad_v[out_v_base + d];
  }
}

template <typename scalar_t>
__global__ void causal_dwconv_input_backward_kernel(
    const scalar_t* __restrict__ grad_preact,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ grad_input,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t K) {
  const int64_t total = B * T * C;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = idx % C;
    const int64_t tmp = idx / C;
    const int64_t t = tmp % T;
    const int64_t b = tmp / T;
    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      const int64_t out_t = t + (K - 1) - k;
      if (out_t >= 0 && out_t < T) {
        const int64_t g_idx = ((b * T + out_t) * C + c);
        acc += to_float(grad_preact[g_idx]) * to_float(weight[c * K + k]);
      }
    }
    grad_input[idx] = from_float<scalar_t>(acc);
  }
}

template <typename scalar_t>
__global__ void causal_dwconv_weight_backward_kernel(
    const scalar_t* __restrict__ grad_preact,
    const scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ grad_weight,
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t K) {
  const int64_t total = C * K;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t c = idx / K;
    const int64_t k = idx % K;
    float acc = 0.0f;
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t t = 0; t < T; ++t) {
        const int64_t tau = t - (K - 1) + k;
        if (tau >= 0 && tau < T) {
          const int64_t g_idx = ((b * T + t) * C + c);
          const int64_t x_idx = ((b * T + tau) * C + c);
          acc += to_float(grad_preact[g_idx]) * to_float(qkv[x_idx]);
        }
      }
    }
    grad_weight[idx] = from_float<scalar_t>(acc);
  }
}

}  // namespace

std::vector<torch::Tensor> packed_qkv_frontend_forward_cuda(
    torch::Tensor qkv,
    torch::Tensor weight,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps) {
  c10::cuda::CUDAGuard device_guard(qkv.device());
  const auto B = qkv.size(0);
  const auto T = qkv.size(1);
  const auto C = qkv.size(2);
  const auto K = weight.size(1);
  const int64_t q_dim = n_heads * head_k_dim;
  const int64_t k_dim = n_heads * head_k_dim;
  const int64_t v_dim = n_heads * head_v_dim;
  TORCH_CHECK(
      C == q_dim + k_dim + v_dim,
      "Packed frontend channel mismatch: got ",
      C,
      " expected ",
      q_dim + k_dim + v_dim);

  auto preact = torch::empty_like(qkv);
  auto q = torch::empty({B, T, n_heads, head_k_dim}, qkv.options());
  auto k = torch::empty({B, T, n_heads, head_k_dim}, qkv.options());
  auto v = torch::empty({B, T, n_heads, head_v_dim}, qkv.options());
  auto inv_opts = qkv.options().dtype(torch::kFloat32);
  auto inv_q = torch::empty({B, T, n_heads}, inv_opts);
  auto inv_k = torch::empty({B, T, n_heads}, inv_opts);

  constexpr int threads = 256;
  const int64_t total = B * T * C;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int launch_blocks = std::min(blocks, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "packed_qkv_frontend_forward_cuda",
      [&] {
        causal_dwconv_preact_forward_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                qkv.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                preact.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
        split_norm_from_preact_kernel<scalar_t>
            <<<B * T * n_heads, 128, sizeof(float) * 128 * 2, stream>>>(
                preact.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                inv_q.data_ptr<float>(),
                inv_k.data_ptr<float>(),
                B * T,
                n_heads,
                head_k_dim,
                head_v_dim,
                C,
                static_cast<float>(eps));
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {q, k, v, preact, inv_q, inv_k};
}

std::vector<torch::Tensor> packed_qkv_conv_forward_cuda(
    torch::Tensor qkv,
    torch::Tensor weight) {
  c10::cuda::CUDAGuard device_guard(qkv.device());
  const auto B = qkv.size(0);
  const auto T = qkv.size(1);
  const auto C = qkv.size(2);
  const auto K = weight.size(1);
  TORCH_CHECK(
      C == weight.size(0),
      "Packed conv channel mismatch: got ",
      C,
      " expected ",
      weight.size(0));

  auto preact = torch::empty_like(qkv);
  auto packed_out = torch::empty_like(qkv);

  constexpr int threads = 256;
  const int64_t total = B * T * C;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int launch_blocks = std::min(blocks, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "packed_qkv_conv_forward_cuda",
      [&] {
        causal_dwconv_preact_forward_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                qkv.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                preact.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
        silu_from_preact_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                preact.data_ptr<scalar_t>(),
                packed_out.data_ptr<scalar_t>(),
                total);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {packed_out, preact};
}

std::vector<torch::Tensor> packed_qkv_frontend_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor qkv,
    torch::Tensor weight,
    torch::Tensor preact,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k) {
  c10::cuda::CUDAGuard device_guard(qkv.device());
  const auto B = qkv.size(0);
  const auto T = qkv.size(1);
  const auto C = qkv.size(2);
  const auto K = weight.size(1);
  const auto H = grad_q.size(2);
  const auto Dk = grad_q.size(3);
  const auto Dv = grad_v.size(3);

  auto grad_preact = torch::empty_like(preact);
  auto grad_input = torch::empty_like(qkv);
  auto grad_weight = torch::empty_like(weight);

  constexpr int threads = 256;
  const int64_t total = B * T * C;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int launch_blocks = std::min(blocks, 65535);
  const int64_t weight_total = C * K;
  const int weight_blocks = static_cast<int>((weight_total + threads - 1) / threads);
  const int weight_launch_blocks = std::min(weight_blocks, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "packed_qkv_frontend_backward_cuda",
      [&] {
        build_grad_preact_kernel<scalar_t>
            <<<B * T * H, 128, sizeof(float) * 128 * 2, stream>>>(
                grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<scalar_t>(),
                grad_v.data_ptr<scalar_t>(),
                preact.data_ptr<scalar_t>(),
                q_norm.data_ptr<scalar_t>(),
                k_norm.data_ptr<scalar_t>(),
                inv_q.data_ptr<float>(),
                inv_k.data_ptr<float>(),
                grad_preact.data_ptr<scalar_t>(),
                B * T,
                H,
                Dk,
                Dv,
                C);
        causal_dwconv_input_backward_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                grad_preact.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
        causal_dwconv_weight_backward_kernel<scalar_t>
            <<<weight_launch_blocks, threads, 0, stream>>>(
                grad_preact.data_ptr<scalar_t>(),
                qkv.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {grad_input, grad_weight};
}

std::vector<torch::Tensor> packed_qkv_conv_backward_cuda(
    torch::Tensor grad_packed_out,
    torch::Tensor qkv,
    torch::Tensor weight,
    torch::Tensor preact) {
  c10::cuda::CUDAGuard device_guard(qkv.device());
  const auto B = qkv.size(0);
  const auto T = qkv.size(1);
  const auto C = qkv.size(2);
  const auto K = weight.size(1);

  auto grad_preact = torch::empty_like(preact);
  auto grad_input = torch::empty_like(qkv);
  auto grad_weight = torch::empty_like(weight);

  constexpr int threads = 256;
  const int64_t total = B * T * C;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const int launch_blocks = std::min(blocks, 65535);
  const int64_t weight_total = C * K;
  const int weight_blocks = static_cast<int>((weight_total + threads - 1) / threads);
  const int weight_launch_blocks = std::min(weight_blocks, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "packed_qkv_conv_backward_cuda",
      [&] {
        silu_grad_from_preact_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                grad_packed_out.data_ptr<scalar_t>(),
                preact.data_ptr<scalar_t>(),
                grad_preact.data_ptr<scalar_t>(),
                total);
        causal_dwconv_input_backward_kernel<scalar_t>
            <<<launch_blocks, threads, 0, stream>>>(
                grad_preact.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
        causal_dwconv_weight_backward_kernel<scalar_t>
            <<<weight_launch_blocks, threads, 0, stream>>>(
                grad_preact.data_ptr<scalar_t>(),
                qkv.data_ptr<scalar_t>(),
                grad_weight.data_ptr<scalar_t>(),
                B,
                T,
                C,
                K);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {grad_input, grad_weight};
}

std::vector<torch::Tensor> packed_qkv_split_l2norm_forward_cuda(
    torch::Tensor packed,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps) {
  c10::cuda::CUDAGuard device_guard(packed.device());
  const auto B = packed.size(0);
  const auto T = packed.size(1);
  const auto C = packed.size(2);
  const int64_t q_dim = n_heads * head_k_dim;
  const int64_t k_dim = n_heads * head_k_dim;
  const int64_t v_dim = n_heads * head_v_dim;
  TORCH_CHECK(
      C == q_dim + k_dim + v_dim,
      "Packed split/l2norm channel mismatch: got ",
      C,
      " expected ",
      q_dim + k_dim + v_dim);

  auto q = torch::empty({B, T, n_heads, head_k_dim}, packed.options());
  auto k = torch::empty({B, T, n_heads, head_k_dim}, packed.options());
  auto v = torch::empty({B, T, n_heads, head_v_dim}, packed.options());
  auto inv_opts = packed.options().dtype(torch::kFloat32);
  auto inv_q = torch::empty({B, T, n_heads}, inv_opts);
  auto inv_k = torch::empty({B, T, n_heads}, inv_opts);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      packed.scalar_type(),
      "packed_qkv_split_l2norm_forward_cuda",
      [&] {
        split_l2norm_from_packed_kernel<scalar_t>
            <<<B * T * n_heads, 128, sizeof(float) * 128 * 2, stream>>>(
                packed.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                inv_q.data_ptr<float>(),
                inv_k.data_ptr<float>(),
                B * T,
                n_heads,
                head_k_dim,
                head_v_dim,
                C,
                static_cast<float>(eps));
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {q, k, v, inv_q, inv_k};
}

torch::Tensor packed_qkv_split_l2norm_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k) {
  c10::cuda::CUDAGuard device_guard(grad_q.device());
  const auto B = grad_q.size(0);
  const auto T = grad_q.size(1);
  const auto H = grad_q.size(2);
  const auto Dk = grad_q.size(3);
  const auto Dv = grad_v.size(3);
  const auto C = H * (2 * Dk + Dv);
  auto grad_packed = torch::empty({B, T, C}, grad_q.options());
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_q.scalar_type(),
      "packed_qkv_split_l2norm_backward_cuda",
      [&] {
        build_grad_packed_from_norm_kernel<scalar_t>
            <<<B * T * H, 128, sizeof(float) * 128 * 2, stream>>>(
                grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<scalar_t>(),
                grad_v.data_ptr<scalar_t>(),
                q_norm.data_ptr<scalar_t>(),
                k_norm.data_ptr<scalar_t>(),
                inv_q.data_ptr<float>(),
                inv_k.data_ptr<float>(),
                grad_packed.data_ptr<scalar_t>(),
                B * T,
                H,
                Dk,
                Dv,
                C);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_packed;
}
