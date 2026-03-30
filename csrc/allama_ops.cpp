#include <torch/extension.h>

namespace {

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight) {
  TORCH_CHECK(x.device() == branch.device(), "x and branch must be on the same device");
  TORCH_CHECK(x.scalar_type() == branch.scalar_type(), "x and branch must have the same dtype");
  TORCH_CHECK(x.sizes() == branch.sizes(), "x and branch must have the same shape");
  TORCH_CHECK(x.dim() >= 2, "expected x to have at least 2 dims");
  TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(scale.numel() == x.size(-1), "scale size must match the last dim of x");
  TORCH_CHECK(weight.numel() == x.size(-1), "weight size must match the last dim of x");
}

torch::Tensor residual_scale_rms_norm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps) {
  auto x2 = x.contiguous().view({-1, x.size(-1)});
  auto branch2 = branch.contiguous().view({-1, branch.size(-1)});
  auto scale_cast = scale.to(x.scalar_type()).view({1, x.size(-1)});
  auto weight_cast = weight.to(x.scalar_type()).view({1, x.size(-1)});
  auto mixed = x2 + branch2 * scale_cast;
  auto inv_rms = (mixed.to(torch::kFloat32).square().mean(-1, true) + eps)
                     .rsqrt()
                     .to(mixed.scalar_type());
  return (mixed * inv_rms * weight_cast).view_as(x);
}

}  // namespace

torch::Tensor residual_scale_rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps);

torch::Tensor residual_scale_rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps) {
  check_inputs(x, branch, scale, weight);
  if (x.is_cuda()) {
    TORCH_CHECK(
        scale.is_cuda() && weight.is_cuda(),
        "scale and weight must be CUDA tensors when x is on CUDA");
    return residual_scale_rms_norm_cuda(x, branch, scale, weight, eps);
  }
  return residual_scale_rms_norm_cpu(x, branch, scale, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "residual_scale_rms_norm",
      &residual_scale_rms_norm,
      "Fused residual add, per-channel scale, and RMSNorm");
}
