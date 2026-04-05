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
__device__ __forceinline__ float to_float_out(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float_out(float x) {
  return static_cast<scalar_t>(x);
}

template <typename scalar_t>
__device__ __forceinline__ float silu_forward_out(scalar_t x) {
  const float xf = to_float_out(x);
  const float sig = 1.0f / (1.0f + expf(-xf));
  return xf * sig;
}

template <typename scalar_t>
__device__ __forceinline__ float silu_backward_out(scalar_t x) {
  const float xf = to_float_out(x);
  const float sig = 1.0f / (1.0f + expf(-xf));
  return sig * (1.0f + xf * (1.0f - sig));
}

template <typename scalar_t>
__global__ void rmsnorm_silu_gate_forward_kernel(
    const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ gate,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ normalized,
    float* __restrict__ inv_rms,
    int64_t rows,
    int64_t D,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float shmem[];
  float sumsq = 0.0f;
  const int64_t base = row * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float ov = to_float_out(o[base + d]);
    sumsq += ov * ov;
  }
  shmem[threadIdx.x] = sumsq;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shmem[threadIdx.x] += shmem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv = rsqrtf(shmem[0] / static_cast<float>(D) + eps);
  if (threadIdx.x == 0) {
    inv_rms[row] = inv;
  }
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float norm = to_float_out(o[base + d]) * inv;
    normalized[base + d] = from_float_out<scalar_t>(norm);
    out[base + d] = from_float_out<scalar_t>(
        norm * silu_forward_out(gate[base + d]));
  }
}

template <typename scalar_t>
__global__ void rmsnorm_silu_gate_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ normalized,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_gate,
    int64_t rows,
    int64_t D) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  extern __shared__ float shmem[];
  float mean_dot = 0.0f;
  const int64_t base = row * D;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float gate_act = silu_forward_out(gate[base + d]);
    const float grad_norm = to_float_out(grad_out[base + d]) * gate_act;
    mean_dot += grad_norm * to_float_out(normalized[base + d]);
  }
  shmem[threadIdx.x] = mean_dot;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shmem[threadIdx.x] += shmem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = shmem[0] / static_cast<float>(D);
  const float inv = inv_rms[row];
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    const float norm = to_float_out(normalized[base + d]);
    const float gout = to_float_out(grad_out[base + d]);
    const float gate_val = to_float_out(gate[base + d]);
    const float sig = 1.0f / (1.0f + expf(-gate_val));
    const float gate_act = gate_val * sig;
    const float grad_norm = gout * gate_act;
    grad_o[base + d] = from_float_out<scalar_t>(inv * (grad_norm - norm * mean));
    grad_gate[base + d] = from_float_out<scalar_t>(
        gout * norm * (sig * (1.0f + gate_val * (1.0f - sig))));
  }
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_silu_gate_forward_cuda(
    torch::Tensor o,
    torch::Tensor gate,
    double eps,
    bool /*fp32_accum*/) {
  c10::cuda::CUDAGuard device_guard(o.device());
  const auto D = o.size(-1);
  const auto rows = o.numel() / D;
  auto out = torch::empty_like(o);
  auto normalized = torch::empty_like(o);
  auto inv_rms = torch::empty({rows}, o.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      o.scalar_type(),
      "rmsnorm_silu_gate_forward_cuda",
      [&] {
        rmsnorm_silu_gate_forward_kernel<scalar_t>
            <<<rows, 128, sizeof(float) * 128, stream>>>(
                o.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                normalized.data_ptr<scalar_t>(),
                inv_rms.data_ptr<float>(),
                rows,
                D,
                static_cast<float>(eps));
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {out, normalized, inv_rms};
}

std::vector<torch::Tensor> rmsnorm_silu_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor normalized,
    torch::Tensor gate,
    torch::Tensor inv_rms) {
  c10::cuda::CUDAGuard device_guard(grad_out.device());
  const auto D = grad_out.size(-1);
  const auto rows = grad_out.numel() / D;
  auto grad_o = torch::empty_like(grad_out);
  auto grad_gate = torch::empty_like(gate);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_out.scalar_type(),
      "rmsnorm_silu_gate_backward_cuda",
      [&] {
        rmsnorm_silu_gate_backward_kernel<scalar_t>
            <<<rows, 128, sizeof(float) * 128, stream>>>(
                grad_out.data_ptr<scalar_t>(),
                normalized.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(),
                inv_rms.data_ptr<float>(),
                grad_o.data_ptr<scalar_t>(),
                grad_gate.data_ptr<scalar_t>(),
                rows,
                D);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return {grad_o, grad_gate};
}
