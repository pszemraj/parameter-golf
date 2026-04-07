#include <torch/extension.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, n) TORCH_CHECK((x).dim() == (n), #x " must have " #n " dims")

std::vector<torch::Tensor> packed_qkv_frontend_forward_cuda(
    torch::Tensor qkv,
    torch::Tensor weight,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps);

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
    torch::Tensor inv_k);

std::vector<torch::Tensor> packed_qkv_conv_forward_cuda(
    torch::Tensor qkv,
    torch::Tensor weight);

std::vector<torch::Tensor> packed_qkv_conv_backward_cuda(
    torch::Tensor grad_packed_out,
    torch::Tensor qkv,
    torch::Tensor weight,
    torch::Tensor preact);

std::vector<torch::Tensor> packed_qkv_conv_input_backward_cuda(
    torch::Tensor grad_packed_out,
    torch::Tensor weight,
    torch::Tensor preact);

std::vector<torch::Tensor> packed_qkv_split_l2norm_forward_cuda(
    torch::Tensor packed,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps);

std::vector<torch::Tensor> preact_silu_split_l2norm_nct_forward_cuda(
    torch::Tensor preact_nct,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps);

torch::Tensor packed_qkv_split_l2norm_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k);

torch::Tensor preact_silu_split_l2norm_nct_backward_cuda(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor preact_nct,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k);

std::vector<torch::Tensor> rmsnorm_silu_gate_forward_cuda(
    torch::Tensor o,
    torch::Tensor gate,
    double eps,
    bool fp32_accum);

std::vector<torch::Tensor> rmsnorm_silu_gate_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor normalized,
    torch::Tensor gate,
    torch::Tensor inv_rms);

std::vector<torch::Tensor> packed_qkv_frontend_forward(
    torch::Tensor qkv,
    torch::Tensor weight,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps) {
  CHECK_CUDA(qkv);
  CHECK_CUDA(weight);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(weight);
  CHECK_DIM(qkv, 3);
  CHECK_DIM(weight, 2);
  TORCH_CHECK(n_heads > 0, "n_heads must be positive");
  TORCH_CHECK(head_k_dim > 0, "head_k_dim must be positive");
  TORCH_CHECK(head_v_dim > 0, "head_v_dim must be positive");
  return packed_qkv_frontend_forward_cuda(
      qkv, weight, n_heads, head_k_dim, head_v_dim, eps);
}

std::vector<torch::Tensor> packed_qkv_conv_forward(
    torch::Tensor qkv,
    torch::Tensor weight) {
  CHECK_CUDA(qkv);
  CHECK_CUDA(weight);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(weight);
  CHECK_DIM(qkv, 3);
  CHECK_DIM(weight, 2);
  return packed_qkv_conv_forward_cuda(qkv, weight);
}

std::vector<torch::Tensor> packed_qkv_frontend_backward(
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
  CHECK_CUDA(grad_q);
  CHECK_CUDA(grad_k);
  CHECK_CUDA(grad_v);
  CHECK_CUDA(qkv);
  CHECK_CUDA(weight);
  CHECK_CUDA(preact);
  CHECK_CUDA(q_norm);
  CHECK_CUDA(k_norm);
  CHECK_CUDA(inv_q);
  CHECK_CUDA(inv_k);
  CHECK_CONTIGUOUS(grad_q);
  CHECK_CONTIGUOUS(grad_k);
  CHECK_CONTIGUOUS(grad_v);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(preact);
  CHECK_CONTIGUOUS(q_norm);
  CHECK_CONTIGUOUS(k_norm);
  CHECK_CONTIGUOUS(inv_q);
  CHECK_CONTIGUOUS(inv_k);
  CHECK_DIM(qkv, 3);
  CHECK_DIM(weight, 2);
  return packed_qkv_frontend_backward_cuda(
      grad_q,
      grad_k,
      grad_v,
      qkv,
      weight,
      preact,
      q_norm,
      k_norm,
      inv_q,
      inv_k);
}

std::vector<torch::Tensor> packed_qkv_conv_backward(
    torch::Tensor grad_packed_out,
    torch::Tensor qkv,
    torch::Tensor weight,
    torch::Tensor preact) {
  CHECK_CUDA(grad_packed_out);
  CHECK_CUDA(qkv);
  CHECK_CUDA(weight);
  CHECK_CUDA(preact);
  CHECK_CONTIGUOUS(grad_packed_out);
  CHECK_CONTIGUOUS(qkv);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(preact);
  CHECK_DIM(grad_packed_out, 3);
  CHECK_DIM(qkv, 3);
  CHECK_DIM(weight, 2);
  CHECK_DIM(preact, 3);
  return packed_qkv_conv_backward_cuda(grad_packed_out, qkv, weight, preact);
}

std::vector<torch::Tensor> packed_qkv_conv_input_backward(
    torch::Tensor grad_packed_out,
    torch::Tensor weight,
    torch::Tensor preact) {
  CHECK_CUDA(grad_packed_out);
  CHECK_CUDA(weight);
  CHECK_CUDA(preact);
  CHECK_CONTIGUOUS(grad_packed_out);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(preact);
  CHECK_DIM(grad_packed_out, 3);
  CHECK_DIM(weight, 2);
  CHECK_DIM(preact, 3);
  return packed_qkv_conv_input_backward_cuda(grad_packed_out, weight, preact);
}

std::vector<torch::Tensor> rmsnorm_silu_gate_forward(
    torch::Tensor o,
    torch::Tensor gate,
    double eps,
    bool fp32_accum) {
  CHECK_CUDA(o);
  CHECK_CUDA(gate);
  CHECK_CONTIGUOUS(o);
  CHECK_CONTIGUOUS(gate);
  TORCH_CHECK(o.sizes() == gate.sizes(), "o and gate must have matching shapes");
  return rmsnorm_silu_gate_forward_cuda(o, gate, eps, fp32_accum);
}

std::vector<torch::Tensor> rmsnorm_silu_gate_backward(
    torch::Tensor grad_out,
    torch::Tensor normalized,
    torch::Tensor gate,
    torch::Tensor inv_rms) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(normalized);
  CHECK_CUDA(gate);
  CHECK_CUDA(inv_rms);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(normalized);
  CHECK_CONTIGUOUS(gate);
  CHECK_CONTIGUOUS(inv_rms);
  TORCH_CHECK(
      grad_out.sizes() == normalized.sizes(),
      "grad_out and normalized must have matching shapes");
  TORCH_CHECK(
      grad_out.sizes() == gate.sizes(),
      "grad_out and gate must have matching shapes");
  return rmsnorm_silu_gate_backward_cuda(grad_out, normalized, gate, inv_rms);
}

std::vector<torch::Tensor> packed_qkv_split_l2norm_forward(
    torch::Tensor packed,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps) {
  CHECK_CUDA(packed);
  CHECK_CONTIGUOUS(packed);
  CHECK_DIM(packed, 3);
  TORCH_CHECK(n_heads > 0, "n_heads must be positive");
  TORCH_CHECK(head_k_dim > 0, "head_k_dim must be positive");
  TORCH_CHECK(head_v_dim > 0, "head_v_dim must be positive");
  return packed_qkv_split_l2norm_forward_cuda(
      packed, n_heads, head_k_dim, head_v_dim, eps);
}

torch::Tensor packed_qkv_split_l2norm_backward(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k) {
  CHECK_CUDA(grad_q);
  CHECK_CUDA(grad_k);
  CHECK_CUDA(grad_v);
  CHECK_CUDA(q_norm);
  CHECK_CUDA(k_norm);
  CHECK_CUDA(inv_q);
  CHECK_CUDA(inv_k);
  CHECK_CONTIGUOUS(grad_q);
  CHECK_CONTIGUOUS(grad_k);
  CHECK_CONTIGUOUS(grad_v);
  CHECK_CONTIGUOUS(q_norm);
  CHECK_CONTIGUOUS(k_norm);
  CHECK_CONTIGUOUS(inv_q);
  CHECK_CONTIGUOUS(inv_k);
  return packed_qkv_split_l2norm_backward_cuda(
      grad_q, grad_k, grad_v, q_norm, k_norm, inv_q, inv_k);
}

std::vector<torch::Tensor> preact_silu_split_l2norm_nct_forward(
    torch::Tensor preact_nct,
    int64_t n_heads,
    int64_t head_k_dim,
    int64_t head_v_dim,
    double eps) {
  CHECK_CUDA(preact_nct);
  CHECK_CONTIGUOUS(preact_nct);
  CHECK_DIM(preact_nct, 3);
  TORCH_CHECK(n_heads > 0, "n_heads must be positive");
  TORCH_CHECK(head_k_dim > 0, "head_k_dim must be positive");
  TORCH_CHECK(head_v_dim > 0, "head_v_dim must be positive");
  return preact_silu_split_l2norm_nct_forward_cuda(
      preact_nct, n_heads, head_k_dim, head_v_dim, eps);
}

torch::Tensor preact_silu_split_l2norm_nct_backward(
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor preact_nct,
    torch::Tensor q_norm,
    torch::Tensor k_norm,
    torch::Tensor inv_q,
    torch::Tensor inv_k) {
  CHECK_CUDA(grad_q);
  CHECK_CUDA(grad_k);
  CHECK_CUDA(grad_v);
  CHECK_CUDA(preact_nct);
  CHECK_CUDA(q_norm);
  CHECK_CUDA(k_norm);
  CHECK_CUDA(inv_q);
  CHECK_CUDA(inv_k);
  CHECK_CONTIGUOUS(grad_q);
  CHECK_CONTIGUOUS(grad_k);
  CHECK_CONTIGUOUS(grad_v);
  CHECK_CONTIGUOUS(preact_nct);
  CHECK_CONTIGUOUS(q_norm);
  CHECK_CONTIGUOUS(k_norm);
  CHECK_CONTIGUOUS(inv_q);
  CHECK_CONTIGUOUS(inv_k);
  return preact_silu_split_l2norm_nct_backward_cuda(
      grad_q, grad_k, grad_v, preact_nct, q_norm, k_norm, inv_q, inv_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "packed_qkv_conv_forward",
      &packed_qkv_conv_forward,
      "HGDN packed qkv conv forward (CUDA)");
  m.def(
      "packed_qkv_conv_backward",
      &packed_qkv_conv_backward,
      "HGDN packed qkv conv backward (CUDA)");
  m.def(
      "packed_qkv_conv_input_backward",
      &packed_qkv_conv_input_backward,
      "HGDN packed qkv conv input backward (CUDA)");
  m.def(
      "packed_qkv_frontend_forward",
      &packed_qkv_frontend_forward,
      "HGDN packed qkv frontend forward (CUDA)");
  m.def(
      "packed_qkv_frontend_backward",
      &packed_qkv_frontend_backward,
      "HGDN packed qkv frontend backward (CUDA)");
  m.def(
      "packed_qkv_split_l2norm_forward",
      &packed_qkv_split_l2norm_forward,
      "HGDN packed qkv split+l2norm forward (CUDA)");
  m.def(
      "packed_qkv_split_l2norm_backward",
      &packed_qkv_split_l2norm_backward,
      "HGDN packed qkv split+l2norm backward (CUDA)");
  m.def(
      "preact_silu_split_l2norm_nct_forward",
      &preact_silu_split_l2norm_nct_forward,
      "HGDN NCT preact->SiLU->split+l2 forward (CUDA)");
  m.def(
      "preact_silu_split_l2norm_nct_backward",
      &preact_silu_split_l2norm_nct_backward,
      "HGDN NCT preact->SiLU->split+l2 backward (CUDA)");
  m.def(
      "rmsnorm_silu_gate_forward",
      &rmsnorm_silu_gate_forward,
      "HGDN RMSNorm*SiLU(gate) forward (CUDA)");
  m.def(
      "rmsnorm_silu_gate_backward",
      &rmsnorm_silu_gate_backward,
      "HGDN RMSNorm*SiLU(gate) backward (CUDA)");
}
