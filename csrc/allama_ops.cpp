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

void check_attention_prep_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  TORCH_CHECK(qkv.dim() == 3, "qkv must be [B, S, D]");
  TORCH_CHECK(q_gain.dim() == 1, "q_gain must be 1D");
  TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2, "cos and sin must be [S, D/2]");
  TORCH_CHECK(head_dim > 0 && head_dim % 2 == 0, "head_dim must be positive and even");
  TORCH_CHECK(num_heads > 0 && num_kv_heads > 0, "head counts must be positive");
  TORCH_CHECK(
      qkv.size(2) == (num_heads + 2 * num_kv_heads) * head_dim,
      "qkv last dim must match head layout");
  TORCH_CHECK(
      q_gain.numel() == num_heads,
      "q_gain length must match num_heads");
  TORCH_CHECK(
      cos.size(0) == qkv.size(1) && sin.size(0) == qkv.size(1),
      "cos/sin sequence dim must match qkv");
  TORCH_CHECK(
      cos.size(1) == head_dim / 2 && sin.size(1) == head_dim / 2,
      "cos/sin rotary dim must match head_dim / 2");
}

void check_attention_prep_backward_inputs(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_v,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  check_attention_prep_inputs(qkv, q_gain, cos, sin, num_heads, num_kv_heads, head_dim);
  TORCH_CHECK(
      grad_q.dim() == 4 && grad_q.size(0) == qkv.size(0) && grad_q.size(1) == qkv.size(1)
          && grad_q.size(2) == num_heads && grad_q.size(3) == head_dim,
      "grad_q must be [B, S, num_heads, head_dim]");
  TORCH_CHECK(
      grad_k.dim() == 4 && grad_k.size(0) == qkv.size(0) && grad_k.size(1) == qkv.size(1)
          && grad_k.size(2) == num_kv_heads && grad_k.size(3) == head_dim,
      "grad_k must be [B, S, num_kv_heads, head_dim]");
  TORCH_CHECK(
      grad_v.sizes() == grad_k.sizes(),
      "grad_v must match grad_k shape");
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attention_prep_bshd_cpu(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  auto qkv_f = qkv.contiguous().to(torch::kFloat32);
  auto q_gain_f = q_gain.contiguous().to(torch::kFloat32);
  auto cos_f = cos.contiguous().to(torch::kFloat32);
  auto sin_f = sin.contiguous().to(torch::kFloat32);
  auto batch_size = qkv.size(0);
  auto seq_len = qkv.size(1);
  auto q_dim = num_heads * head_dim;
  auto kv_dim = num_kv_heads * head_dim;
  auto q = qkv_f.narrow(-1, 0, q_dim).view({batch_size, seq_len, num_heads, head_dim});
  auto k = qkv_f.narrow(-1, q_dim, kv_dim).view({batch_size, seq_len, num_kv_heads, head_dim});
  auto v = qkv_f.narrow(-1, q_dim + kv_dim, kv_dim)
               .view({batch_size, seq_len, num_kv_heads, head_dim});
  auto q_rms = (q.square().mean(-1, true) + 1e-5).rsqrt();
  auto k_rms = (k.square().mean(-1, true) + 1e-5).rsqrt();
  q = q * q_rms;
  k = k * k_rms;
  auto half = head_dim / 2;
  auto cos_view = cos_f.view({1, seq_len, 1, half});
  auto sin_view = sin_f.view({1, seq_len, 1, half});
  auto q1 = q.slice(-1, 0, half);
  auto q2 = q.slice(-1, half, head_dim);
  auto k1 = k.slice(-1, 0, half);
  auto k2 = k.slice(-1, half, head_dim);
  auto q_rot = torch::cat({q1 * cos_view + q2 * sin_view, q2 * cos_view - q1 * sin_view}, -1);
  auto k_rot = torch::cat({k1 * cos_view + k2 * sin_view, k2 * cos_view - k1 * sin_view}, -1);
  q_rot = q_rot * q_gain_f.view({1, 1, num_heads, 1});
  return {
      q_rot.to(qkv.scalar_type()),
      k_rot.to(qkv.scalar_type()),
      v.to(qkv.scalar_type()),
  };
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attention_prep_bshd_cuda(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim);

std::tuple<torch::Tensor, torch::Tensor> attention_prep_bshd_backward_cuda(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_v,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim);

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attention_prep_bshd(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  check_attention_prep_inputs(qkv, q_gain, cos, sin, num_heads, num_kv_heads, head_dim);
  if (qkv.is_cuda()) {
    TORCH_CHECK(
        q_gain.is_cuda() && cos.is_cuda() && sin.is_cuda(),
        "q_gain, cos, and sin must be CUDA tensors when qkv is on CUDA");
    return attention_prep_bshd_cuda(qkv, q_gain, cos, sin, num_heads, num_kv_heads, head_dim);
  }
  return attention_prep_bshd_cpu(qkv, q_gain, cos, sin, num_heads, num_kv_heads, head_dim);
}

std::tuple<torch::Tensor, torch::Tensor> attention_prep_bshd_backward(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_v,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  check_attention_prep_backward_inputs(
      qkv, q_gain, cos, sin, grad_q, grad_k, grad_v, num_heads, num_kv_heads, head_dim);
  TORCH_CHECK(qkv.is_cuda(), "attention_prep_bshd_backward is CUDA-only");
  TORCH_CHECK(
      q_gain.is_cuda() && cos.is_cuda() && sin.is_cuda()
          && grad_q.is_cuda() && grad_k.is_cuda() && grad_v.is_cuda(),
      "attention_prep_bshd_backward inputs must be CUDA tensors");
  return attention_prep_bshd_backward_cuda(
      qkv, q_gain, cos, sin, grad_q, grad_k, grad_v, num_heads, num_kv_heads, head_dim);
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
  m.def(
      "attention_prep_bshd",
      &attention_prep_bshd,
      "C++/CUDA fused FA2 attention prep forward");
  m.def(
      "attention_prep_bshd_backward",
      &attention_prep_bshd_backward,
      "C++/CUDA fused FA2 attention prep backward");
}
