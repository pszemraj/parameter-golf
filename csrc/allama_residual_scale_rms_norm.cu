#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void residual_scale_rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ branch,
    const float* __restrict__ scale,
    const float* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols,
    float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared_sum[];
  float local_sum = 0.0f;
  int64_t base = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float x_val = static_cast<float>(x[base + col]);
    float branch_val = static_cast<float>(branch[base + col]);
    float mixed = x_val + scale[col] * branch_val;
    local_sum += mixed * mixed;
  }

  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sum[0] / static_cast<float>(cols) + eps);
  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float x_val = static_cast<float>(x[base + col]);
    float branch_val = static_cast<float>(branch[base + col]);
    float mixed = x_val + scale[col] * branch_val;
    out[base + col] = static_cast<scalar_t>(mixed * inv_rms * weight[col]);
  }
}

}  // namespace

torch::Tensor residual_scale_rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps) {
  auto x2 = x.contiguous().view({-1, x.size(-1)});
  auto branch2 = branch.contiguous().view({-1, branch.size(-1)});
  auto scale_f = scale.contiguous().to(torch::kFloat32);
  auto weight_f = weight.contiguous().to(torch::kFloat32);
  auto out2 = torch::empty_like(x2);

  int64_t rows = x2.size(0);
  int64_t cols = x2.size(1);
  int threads = 256;
  size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      x2.scalar_type(),
      "residual_scale_rms_norm_cuda",
      [&] {
        residual_scale_rms_norm_kernel<scalar_t><<<
            rows,
            threads,
            shared_mem,
            stream>>>(
            x2.data_ptr<scalar_t>(),
            branch2.data_ptr<scalar_t>(),
            scale_f.data_ptr<float>(),
            weight_f.data_ptr<float>(),
            out2.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out2.view_as(x);
}
