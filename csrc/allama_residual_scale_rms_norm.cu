#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <tuple>

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

template <typename scalar_t>
__global__ void residual_scale_rms_norm_pair_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ branch,
    const float* __restrict__ scale,
    const float* __restrict__ weight,
    scalar_t* __restrict__ mixed_out,
    scalar_t* __restrict__ normed_out,
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
    mixed_out[base + col] = static_cast<scalar_t>(mixed);
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
    float mixed = static_cast<float>(mixed_out[base + col]);
    normed_out[base + col] = static_cast<scalar_t>(mixed * inv_rms * weight[col]);
  }
}

template <typename scalar_t>
__global__ void residual_scale_rms_norm_pair_backward_kernel(
    const scalar_t* __restrict__ mixed,
    const scalar_t* __restrict__ branch,
    const float* __restrict__ scale,
    const float* __restrict__ weight,
    const scalar_t* __restrict__ grad_mixed_out,
    const scalar_t* __restrict__ grad_normed_out,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_branch,
    float* __restrict__ grad_scale,
    float* __restrict__ grad_weight,
    int64_t rows,
    int64_t cols,
    float eps) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared[];
  float* shared_sumsq = shared;
  float* shared_dot = shared + blockDim.x;

  float local_sumsq = 0.0f;
  float local_dot = 0.0f;
  int64_t base = row * cols;

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float mixed_val = static_cast<float>(mixed[base + col]);
    float grad_normed_val = static_cast<float>(grad_normed_out[base + col]);
    local_sumsq += mixed_val * mixed_val;
    local_dot += grad_normed_val * weight[col] * mixed_val;
  }

  shared_sumsq[threadIdx.x] = local_sumsq;
  shared_dot[threadIdx.x] = local_dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + stride];
      shared_dot[threadIdx.x] += shared_dot[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared_sumsq[0] / static_cast<float>(cols) + eps);
  float coeff = shared_dot[0] * inv_rms * inv_rms * inv_rms
      / static_cast<float>(cols);

  for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float mixed_val = static_cast<float>(mixed[base + col]);
    float branch_val = static_cast<float>(branch[base + col]);
    float grad_mixed_val = static_cast<float>(grad_mixed_out[base + col]);
    float grad_normed_val = static_cast<float>(grad_normed_out[base + col]);
    float grad_h = grad_normed_val * weight[col];
    float grad_total =
        grad_mixed_val + inv_rms * grad_h - mixed_val * coeff;
    grad_x[base + col] = static_cast<scalar_t>(grad_total);
    grad_branch[base + col] = static_cast<scalar_t>(grad_total * scale[col]);
    atomicAdd(&grad_scale[col], grad_total * branch_val);
    atomicAdd(&grad_weight[col], grad_normed_val * mixed_val * inv_rms);
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

std::tuple<torch::Tensor, torch::Tensor> residual_scale_rms_norm_pair_cuda(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps) {
  auto x2 = x.contiguous().view({-1, x.size(-1)});
  auto branch2 = branch.contiguous().view({-1, branch.size(-1)});
  auto scale_f = scale.contiguous().to(torch::kFloat32);
  auto weight_f = weight.contiguous().to(torch::kFloat32);
  auto mixed2 = torch::empty_like(x2);
  auto normed2 = torch::empty_like(x2);

  int64_t rows = x2.size(0);
  int64_t cols = x2.size(1);
  int threads = 256;
  size_t shared_mem = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      x2.scalar_type(),
      "residual_scale_rms_norm_pair_cuda",
      [&] {
        residual_scale_rms_norm_pair_kernel<scalar_t><<<
            rows,
            threads,
            shared_mem,
            stream>>>(
            x2.data_ptr<scalar_t>(),
            branch2.data_ptr<scalar_t>(),
            scale_f.data_ptr<float>(),
            weight_f.data_ptr<float>(),
            mixed2.data_ptr<scalar_t>(),
            normed2.data_ptr<scalar_t>(),
            rows,
            cols,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {mixed2.view_as(x), normed2.view_as(x)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
residual_scale_rms_norm_pair_backward_cuda(
    const torch::Tensor& mixed,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    const torch::Tensor& grad_mixed_out,
    const torch::Tensor& grad_normed_out,
    double eps) {
  auto mixed2 = mixed.contiguous().view({-1, mixed.size(-1)});
  auto branch2 = branch.contiguous().view({-1, branch.size(-1)});
  auto grad_mixed2 =
      grad_mixed_out.contiguous().view({-1, grad_mixed_out.size(-1)});
  auto grad_normed2 =
      grad_normed_out.contiguous().view({-1, grad_normed_out.size(-1)});
  auto scale_f = scale.contiguous().to(torch::kFloat32);
  auto weight_f = weight.contiguous().to(torch::kFloat32);
  auto grad_x2 = torch::empty_like(mixed2);
  auto grad_branch2 = torch::empty_like(branch2);
  auto grad_scale = torch::zeros_like(scale_f);
  auto grad_weight = torch::zeros_like(weight_f);

  int64_t rows = mixed2.size(0);
  int64_t cols = mixed2.size(1);
  int threads = 256;
  size_t shared_mem = static_cast<size_t>(threads) * sizeof(float) * 2;
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      mixed2.scalar_type(),
      "residual_scale_rms_norm_pair_backward_cuda",
      [&] {
        residual_scale_rms_norm_pair_backward_kernel<scalar_t><<<
            rows,
            threads,
            shared_mem,
            stream>>>(
            mixed2.data_ptr<scalar_t>(),
            branch2.data_ptr<scalar_t>(),
            scale_f.data_ptr<float>(),
            weight_f.data_ptr<float>(),
            grad_mixed2.data_ptr<scalar_t>(),
            grad_normed2.data_ptr<scalar_t>(),
            grad_x2.data_ptr<scalar_t>(),
            grad_branch2.data_ptr<scalar_t>(),
            grad_scale.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            rows,
            cols,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {
      grad_x2.view_as(mixed),
      grad_branch2.view_as(branch),
      grad_scale.view_as(scale),
      grad_weight.view_as(weight),
  };
}
