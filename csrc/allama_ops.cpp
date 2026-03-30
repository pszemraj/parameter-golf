#include <torch/extension.h>
#include <tuple>

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

void check_grad_inputs(
    const torch::Tensor& mixed,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    const torch::Tensor& grad_mixed_out,
    const torch::Tensor& grad_normed_out) {
  check_inputs(mixed, branch, scale, weight);
  TORCH_CHECK(
      grad_mixed_out.device() == mixed.device() && grad_normed_out.device() == mixed.device(),
      "backward grads must be on the same device as mixed");
  TORCH_CHECK(
      grad_mixed_out.scalar_type() == mixed.scalar_type()
          && grad_normed_out.scalar_type() == mixed.scalar_type(),
      "backward grads must match mixed dtype");
  TORCH_CHECK(
      grad_mixed_out.sizes() == mixed.sizes() && grad_normed_out.sizes() == mixed.sizes(),
      "backward grads must match mixed shape");
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

std::tuple<torch::Tensor, torch::Tensor> residual_scale_rms_norm_pair_cpu(
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
  auto normed = mixed * inv_rms * weight_cast;
  return {mixed.view_as(x), normed.view_as(x)};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
residual_scale_rms_norm_pair_backward_cpu(
    const torch::Tensor& mixed,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    const torch::Tensor& grad_mixed_out,
    const torch::Tensor& grad_normed_out,
    double eps) {
  auto mixed2 = mixed.contiguous().view({-1, mixed.size(-1)});
  auto branch2 = branch.contiguous().view({-1, branch.size(-1)});
  auto grad_mixed2 = grad_mixed_out.contiguous().view({-1, grad_mixed_out.size(-1)});
  auto grad_normed2 =
      grad_normed_out.contiguous().view({-1, grad_normed_out.size(-1)});
  auto scale_f = scale.contiguous().to(torch::kFloat32).view({1, mixed.size(-1)});
  auto weight_f = weight.contiguous().to(torch::kFloat32).view({1, mixed.size(-1)});

  auto mixed_f = mixed2.to(torch::kFloat32);
  auto branch_f = branch2.to(torch::kFloat32);
  auto grad_total_f = grad_mixed2.to(torch::kFloat32);
  auto grad_normed_f = grad_normed2.to(torch::kFloat32);
  auto inv_rms_f = (mixed_f.square().mean(-1, true) + eps).rsqrt();
  auto grad_h_f = grad_normed_f * weight_f;
  auto dot_f = (grad_h_f * mixed_f).sum(-1, true);
  grad_total_f += inv_rms_f * grad_h_f
      - inv_rms_f.pow(3) * mixed_f * dot_f / static_cast<double>(mixed.size(-1));

  auto grad_x = grad_total_f.to(mixed.scalar_type()).view_as(mixed);
  auto grad_branch = (grad_total_f * scale_f).to(branch.scalar_type()).view_as(branch);
  auto grad_scale =
      (grad_total_f * branch_f).sum(0).to(scale.scalar_type()).view_as(scale);
  auto grad_weight =
      (grad_normed_f * mixed_f * inv_rms_f).sum(0).to(weight.scalar_type()).view_as(weight);
  return {grad_x, grad_branch, grad_scale, grad_weight};
}

}  // namespace

torch::Tensor residual_scale_rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> residual_scale_rms_norm_pair_cuda(
    const torch::Tensor& x,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
residual_scale_rms_norm_pair_backward_cuda(
    const torch::Tensor& mixed,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    const torch::Tensor& grad_mixed_out,
    const torch::Tensor& grad_normed_out,
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

std::tuple<torch::Tensor, torch::Tensor> residual_scale_rms_norm_pair(
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
    return residual_scale_rms_norm_pair_cuda(x, branch, scale, weight, eps);
  }
  return residual_scale_rms_norm_pair_cpu(x, branch, scale, weight, eps);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
residual_scale_rms_norm_pair_backward(
    const torch::Tensor& mixed,
    const torch::Tensor& branch,
    const torch::Tensor& scale,
    const torch::Tensor& weight,
    const torch::Tensor& grad_mixed_out,
    const torch::Tensor& grad_normed_out,
    double eps) {
  check_grad_inputs(
      mixed, branch, scale, weight, grad_mixed_out, grad_normed_out);
  if (mixed.is_cuda()) {
    TORCH_CHECK(
        scale.is_cuda() && weight.is_cuda(),
        "scale and weight must be CUDA tensors when mixed is on CUDA");
    return residual_scale_rms_norm_pair_backward_cuda(
        mixed, branch, scale, weight, grad_mixed_out, grad_normed_out, eps);
  }
  return residual_scale_rms_norm_pair_backward_cpu(
      mixed, branch, scale, weight, grad_mixed_out, grad_normed_out, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "residual_scale_rms_norm",
      &residual_scale_rms_norm,
      "Fused residual add, per-channel scale, and RMSNorm");
  m.def(
      "residual_scale_rms_norm_pair",
      &residual_scale_rms_norm_pair,
      "Fused residual add, per-channel scale, and RMSNorm with mixed output");
  m.def(
      "residual_scale_rms_norm_pair_backward",
      &residual_scale_rms_norm_pair_backward,
      "Backward for fused residual add, per-channel scale, and RMSNorm pair op");
}
