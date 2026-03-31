#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <tuple>

namespace {

template <typename scalar_t>
__device__ inline float to_float(scalar_t value) {
  return static_cast<float>(value);
}

template <typename scalar_t>
__device__ inline scalar_t from_float(float value) {
  return static_cast<scalar_t>(value);
}

template <>
__device__ inline c10::Half from_float<c10::Half>(float value) {
  return c10::Half(value);
}

template <>
__device__ inline c10::BFloat16 from_float<c10::BFloat16>(float value) {
  return c10::BFloat16(value);
}

template <int THREADS>
__device__ inline float block_reduce_sum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename scalar_t, int THREADS>
__global__ void attention_prep_bshd_forward_kernel(
    const scalar_t* __restrict__ qkv,
    const float* __restrict__ q_gain,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ q_out,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int q_dim,
    int kv_dim) {
  int token_idx = static_cast<int>(blockIdx.x);
  int head_idx = static_cast<int>(blockIdx.y);
  int b = token_idx / seq_len;
  int t = token_idx % seq_len;
  if (b >= batch_size) {
    return;
  }

  int half = head_dim / 2;
  int lane = static_cast<int>(threadIdx.x);
  extern __shared__ float shared[];

  const int token_offset = (b * seq_len + t) * (q_dim + 2 * kv_dim);
  const int rotary_offset = t * half;

  if (head_idx < num_heads) {
    float local_sum = 0.0f;
    float q1 = 0.0f;
    float q2 = 0.0f;
    float cos_val = 0.0f;
    float sin_val = 0.0f;
    if (lane < half) {
      const int base = token_offset + head_idx * head_dim;
      q1 = to_float(qkv[base + lane]);
      q2 = to_float(qkv[base + half + lane]);
      cos_val = to_float(cos[rotary_offset + lane]);
      sin_val = to_float(sin[rotary_offset + lane]);
      local_sum = q1 * q1 + q2 * q2;
    }
    const float sumsq = block_reduce_sum<THREADS>(local_sum, shared);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(head_dim) + 1e-5f);
    if (lane < half) {
      const float q1_norm = q1 * inv_rms;
      const float q2_norm = q2 * inv_rms;
      const float rot_first = q1_norm * cos_val + q2_norm * sin_val;
      const float rot_second = q2_norm * cos_val - q1_norm * sin_val;
      const float gain = q_gain[head_idx];
      const int out_base = ((b * seq_len + t) * num_heads + head_idx) * head_dim;
      q_out[out_base + lane] = from_float<scalar_t>(rot_first * gain);
      q_out[out_base + half + lane] = from_float<scalar_t>(rot_second * gain);
    }
    return;
  }

  const int kv_head = head_idx - num_heads;
  if (kv_head >= num_kv_heads) {
    return;
  }

  float local_sum = 0.0f;
  float k1 = 0.0f;
  float k2 = 0.0f;
  float cos_val = 0.0f;
  float sin_val = 0.0f;
  if (lane < half) {
    const int k_base = token_offset + q_dim + kv_head * head_dim;
    k1 = to_float(qkv[k_base + lane]);
    k2 = to_float(qkv[k_base + half + lane]);
    cos_val = to_float(cos[rotary_offset + lane]);
    sin_val = to_float(sin[rotary_offset + lane]);
    local_sum = k1 * k1 + k2 * k2;
  }
  const float sumsq = block_reduce_sum<THREADS>(local_sum, shared);
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(head_dim) + 1e-5f);
  if (lane < half) {
    const float k1_norm = k1 * inv_rms;
    const float k2_norm = k2 * inv_rms;
    const float rot_first = k1_norm * cos_val + k2_norm * sin_val;
    const float rot_second = k2_norm * cos_val - k1_norm * sin_val;
    const int out_base = ((b * seq_len + t) * num_kv_heads + kv_head) * head_dim;
    k_out[out_base + lane] = from_float<scalar_t>(rot_first);
    k_out[out_base + half + lane] = from_float<scalar_t>(rot_second);

    const int v_base = token_offset + q_dim + kv_dim + kv_head * head_dim;
    v_out[out_base + lane] = qkv[v_base + lane];
    v_out[out_base + half + lane] = qkv[v_base + half + lane];
  }
}

template <typename scalar_t, int THREADS>
__global__ void attention_prep_bshd_backward_q_kernel(
    const scalar_t* __restrict__ qkv,
    const float* __restrict__ q_gain,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ grad_q,
    scalar_t* __restrict__ grad_qkv,
    float* __restrict__ grad_q_gain,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int qkv_dim) {
  int token_idx = static_cast<int>(blockIdx.x);
  int head_idx = static_cast<int>(blockIdx.y);
  int b = token_idx / seq_len;
  int t = token_idx % seq_len;
  if (b >= batch_size || head_idx >= num_heads) {
    return;
  }

  int half = head_dim / 2;
  int lane = static_cast<int>(threadIdx.x);
  extern __shared__ float shared[];
  float* shared_sumsq = shared;
  float* shared_dot = shared + THREADS;
  float* shared_gain = shared + 2 * THREADS;

  float local_sumsq = 0.0f;
  float local_dot = 0.0f;
  float local_gain = 0.0f;
  float q1_norm = 0.0f;
  float q2_norm = 0.0f;
  float grad_q1_norm = 0.0f;
  float grad_q2_norm = 0.0f;
  if (lane < half) {
    const int token_offset = (b * seq_len + t) * qkv_dim;
    const int rotary_offset = t * half;
    const int q_base = token_offset + head_idx * head_dim;
    const float q1 = to_float(qkv[q_base + lane]);
    const float q2 = to_float(qkv[q_base + half + lane]);
    const float cos_val = to_float(cos[rotary_offset + lane]);
    const float sin_val = to_float(sin[rotary_offset + lane]);
    local_sumsq = q1 * q1 + q2 * q2;

    shared_sumsq[lane] = local_sumsq;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        shared_sumsq[lane] += shared_sumsq[lane + stride];
      }
      __syncthreads();
    }
    const float inv_rms = rsqrtf(shared_sumsq[0] / static_cast<float>(head_dim) + 1e-5f);
    q1_norm = q1 * inv_rms;
    q2_norm = q2 * inv_rms;
    const float rot_first = q1_norm * cos_val + q2_norm * sin_val;
    const float rot_second = q2_norm * cos_val - q1_norm * sin_val;
    const float gain = q_gain[head_idx];
    const int grad_q_base =
        ((b * seq_len + t) * num_heads + head_idx) * head_dim;
    const float grad_q1 = to_float(grad_q[grad_q_base + lane]);
    const float grad_q2 = to_float(grad_q[grad_q_base + half + lane]);
    const float grad_rot_first = grad_q1 * gain;
    const float grad_rot_second = grad_q2 * gain;
    grad_q1_norm = grad_rot_first * cos_val - grad_rot_second * sin_val;
    grad_q2_norm = grad_rot_first * sin_val + grad_rot_second * cos_val;
    local_dot = grad_q1_norm * q1_norm + grad_q2_norm * q2_norm;
    local_gain = grad_q1 * rot_first + grad_q2 * rot_second;

    shared_dot[lane] = local_dot;
    shared_gain[lane] = local_gain;
  } else {
    shared_sumsq[lane] = 0.0f;
    shared_dot[lane] = 0.0f;
    shared_gain[lane] = 0.0f;
  }
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared_dot[lane] += shared_dot[lane + stride];
      shared_gain[lane] += shared_gain[lane + stride];
    }
    __syncthreads();
  }

  if (lane < half) {
    const int token_offset = (b * seq_len + t) * qkv_dim;
    const int q_base = token_offset + head_idx * head_dim;
    const float inv_rms = rsqrtf(shared_sumsq[0] / static_cast<float>(head_dim) + 1e-5f);
    const float dot = shared_dot[0] / static_cast<float>(head_dim);
    const float grad_q1_orig = (grad_q1_norm - q1_norm * dot) * inv_rms;
    const float grad_q2_orig = (grad_q2_norm - q2_norm * dot) * inv_rms;
    grad_qkv[q_base + lane] = from_float<scalar_t>(grad_q1_orig);
    grad_qkv[q_base + half + lane] = from_float<scalar_t>(grad_q2_orig);
    if (lane == 0) {
      atomicAdd(&grad_q_gain[head_idx], shared_gain[0]);
    }
  }
}

template <typename scalar_t, int THREADS>
__global__ void attention_prep_bshd_backward_kv_kernel(
    const scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const scalar_t* __restrict__ grad_k,
    const scalar_t* __restrict__ grad_v,
    scalar_t* __restrict__ grad_qkv,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int q_dim,
    int kv_dim,
    int qkv_dim) {
  int token_idx = static_cast<int>(blockIdx.x);
  int kv_head = static_cast<int>(blockIdx.y);
  int b = token_idx / seq_len;
  int t = token_idx % seq_len;
  if (b >= batch_size || kv_head >= num_kv_heads) {
    return;
  }

  int half = head_dim / 2;
  int lane = static_cast<int>(threadIdx.x);
  extern __shared__ float shared[];
  float* shared_sumsq = shared;
  float* shared_dot = shared + THREADS;

  float local_sumsq = 0.0f;
  float local_dot = 0.0f;
  float k1_norm = 0.0f;
  float k2_norm = 0.0f;
  float grad_k1_norm = 0.0f;
  float grad_k2_norm = 0.0f;
  if (lane < half) {
    const int token_offset = (b * seq_len + t) * qkv_dim;
    const int rotary_offset = t * half;
    const int k_base = token_offset + q_dim + kv_head * head_dim;
    const float k1 = to_float(qkv[k_base + lane]);
    const float k2 = to_float(qkv[k_base + half + lane]);
    const float cos_val = to_float(cos[rotary_offset + lane]);
    const float sin_val = to_float(sin[rotary_offset + lane]);
    local_sumsq = k1 * k1 + k2 * k2;

    shared_sumsq[lane] = local_sumsq;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
      if (lane < stride) {
        shared_sumsq[lane] += shared_sumsq[lane + stride];
      }
      __syncthreads();
    }
    const float inv_rms = rsqrtf(shared_sumsq[0] / static_cast<float>(head_dim) + 1e-5f);
    k1_norm = k1 * inv_rms;
    k2_norm = k2 * inv_rms;
    const int grad_k_base =
        ((b * seq_len + t) * num_kv_heads + kv_head) * head_dim;
    const float grad_k1 = to_float(grad_k[grad_k_base + lane]);
    const float grad_k2 = to_float(grad_k[grad_k_base + half + lane]);
    grad_k1_norm = grad_k1 * cos_val - grad_k2 * sin_val;
    grad_k2_norm = grad_k1 * sin_val + grad_k2 * cos_val;
    local_dot = grad_k1_norm * k1_norm + grad_k2_norm * k2_norm;
    shared_dot[lane] = local_dot;
  } else {
    shared_sumsq[lane] = 0.0f;
    shared_dot[lane] = 0.0f;
  }
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared_dot[lane] += shared_dot[lane + stride];
    }
    __syncthreads();
  }

  if (lane < half) {
    const int token_offset = (b * seq_len + t) * qkv_dim;
    const int k_base = token_offset + q_dim + kv_head * head_dim;
    const float inv_rms = rsqrtf(shared_sumsq[0] / static_cast<float>(head_dim) + 1e-5f);
    const float dot = shared_dot[0] / static_cast<float>(head_dim);
    const float grad_k1_orig = (grad_k1_norm - k1_norm * dot) * inv_rms;
    const float grad_k2_orig = (grad_k2_norm - k2_norm * dot) * inv_rms;
    grad_qkv[k_base + lane] = from_float<scalar_t>(grad_k1_orig);
    grad_qkv[k_base + half + lane] = from_float<scalar_t>(grad_k2_orig);

    const int grad_v_base =
        ((b * seq_len + t) * num_kv_heads + kv_head) * head_dim;
    const int v_base = token_offset + q_dim + kv_dim + kv_head * head_dim;
    grad_qkv[v_base + lane] = grad_v[grad_v_base + lane];
    grad_qkv[v_base + half + lane] = grad_v[grad_v_base + half + lane];
  }
}

template <typename scalar_t>
void launch_attention_prep_bshd_forward(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int q_dim,
    int kv_dim,
    cudaStream_t stream) {
  const int half = head_dim / 2;
  dim3 grid(batch_size * seq_len, num_heads + num_kv_heads);
  if (half <= 32) {
    attention_prep_bshd_forward_kernel<scalar_t, 32><<<
        grid, 32, 32 * sizeof(float), stream>>>(
        qkv.data_ptr<scalar_t>(),
        q_gain.data_ptr<float>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim);
    return;
  }
  if (half <= 64) {
    attention_prep_bshd_forward_kernel<scalar_t, 64><<<
        grid, 64, 64 * sizeof(float), stream>>>(
        qkv.data_ptr<scalar_t>(),
        q_gain.data_ptr<float>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        q_out.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim);
    return;
  }
  attention_prep_bshd_forward_kernel<scalar_t, 128><<<
      grid, 128, 128 * sizeof(float), stream>>>(
      qkv.data_ptr<scalar_t>(),
      q_gain.data_ptr<float>(),
      cos.data_ptr<scalar_t>(),
      sin.data_ptr<scalar_t>(),
      q_out.data_ptr<scalar_t>(),
      k_out.data_ptr<scalar_t>(),
      v_out.data_ptr<scalar_t>(),
      batch_size,
      seq_len,
      num_heads,
      num_kv_heads,
      head_dim,
      q_dim,
      kv_dim);
}

template <typename scalar_t>
void launch_attention_prep_bshd_backward(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& grad_q,
    const torch::Tensor& grad_k,
    const torch::Tensor& grad_v,
    torch::Tensor& grad_qkv,
    torch::Tensor& grad_q_gain,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int q_dim,
    int kv_dim,
    int qkv_dim,
    cudaStream_t stream) {
  const int half = head_dim / 2;
  dim3 q_grid(batch_size * seq_len, num_heads);
  dim3 kv_grid(batch_size * seq_len, num_kv_heads);
  if (half <= 32) {
    attention_prep_bshd_backward_q_kernel<scalar_t, 32><<<
        q_grid, 32, 32 * sizeof(float) * 3, stream>>>(
        qkv.data_ptr<scalar_t>(),
        q_gain.data_ptr<float>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        grad_q.data_ptr<scalar_t>(),
        grad_qkv.data_ptr<scalar_t>(),
        grad_q_gain.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        qkv_dim);
    attention_prep_bshd_backward_kv_kernel<scalar_t, 32><<<
        kv_grid, 32, 32 * sizeof(float) * 2, stream>>>(
        qkv.data_ptr<scalar_t>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        grad_k.data_ptr<scalar_t>(),
        grad_v.data_ptr<scalar_t>(),
        grad_qkv.data_ptr<scalar_t>(),
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim,
        qkv_dim);
    return;
  }
  if (half <= 64) {
    attention_prep_bshd_backward_q_kernel<scalar_t, 64><<<
        q_grid, 64, 64 * sizeof(float) * 3, stream>>>(
        qkv.data_ptr<scalar_t>(),
        q_gain.data_ptr<float>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        grad_q.data_ptr<scalar_t>(),
        grad_qkv.data_ptr<scalar_t>(),
        grad_q_gain.data_ptr<float>(),
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        qkv_dim);
    attention_prep_bshd_backward_kv_kernel<scalar_t, 64><<<
        kv_grid, 64, 64 * sizeof(float) * 2, stream>>>(
        qkv.data_ptr<scalar_t>(),
        cos.data_ptr<scalar_t>(),
        sin.data_ptr<scalar_t>(),
        grad_k.data_ptr<scalar_t>(),
        grad_v.data_ptr<scalar_t>(),
        grad_qkv.data_ptr<scalar_t>(),
        batch_size,
        seq_len,
        num_kv_heads,
        head_dim,
        q_dim,
        kv_dim,
        qkv_dim);
    return;
  }
  attention_prep_bshd_backward_q_kernel<scalar_t, 128><<<
      q_grid, 128, 128 * sizeof(float) * 3, stream>>>(
      qkv.data_ptr<scalar_t>(),
      q_gain.data_ptr<float>(),
      cos.data_ptr<scalar_t>(),
      sin.data_ptr<scalar_t>(),
      grad_q.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(),
      grad_q_gain.data_ptr<float>(),
      batch_size,
      seq_len,
      num_heads,
      head_dim,
      qkv_dim);
  attention_prep_bshd_backward_kv_kernel<scalar_t, 128><<<
      kv_grid, 128, 128 * sizeof(float) * 2, stream>>>(
      qkv.data_ptr<scalar_t>(),
      cos.data_ptr<scalar_t>(),
      sin.data_ptr<scalar_t>(),
      grad_k.data_ptr<scalar_t>(),
      grad_v.data_ptr<scalar_t>(),
      grad_qkv.data_ptr<scalar_t>(),
      batch_size,
      seq_len,
      num_kv_heads,
      head_dim,
      q_dim,
      kv_dim,
      qkv_dim);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attention_prep_bshd_cuda(
    const torch::Tensor& qkv,
    const torch::Tensor& q_gain,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  auto qkv_c = qkv.contiguous();
  auto q_gain_f = q_gain.contiguous().to(torch::kFloat32);
  auto cos_c = cos.contiguous().to(qkv.scalar_type());
  auto sin_c = sin.contiguous().to(qkv.scalar_type());

  const int batch_size = static_cast<int>(qkv_c.size(0));
  const int seq_len = static_cast<int>(qkv_c.size(1));
  const int q_dim = static_cast<int>(num_heads * head_dim);
  const int kv_dim = static_cast<int>(num_kv_heads * head_dim);
  auto q = torch::empty(
      {batch_size, seq_len, num_heads, head_dim},
      qkv.options());
  auto k = torch::empty(
      {batch_size, seq_len, num_kv_heads, head_dim},
      qkv.options());
  auto v = torch::empty_like(k);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      qkv_c.scalar_type(),
      "attention_prep_bshd_cuda",
      [&] {
        launch_attention_prep_bshd_forward<scalar_t>(
            qkv_c,
            q_gain_f,
            cos_c,
            sin_c,
            q,
            k,
            v,
            batch_size,
            seq_len,
            static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            q_dim,
            kv_dim,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q, k, v};
}

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
    int64_t head_dim) {
  auto qkv_c = qkv.contiguous();
  auto q_gain_f = q_gain.contiguous().to(torch::kFloat32);
  auto cos_c = cos.contiguous().to(qkv.scalar_type());
  auto sin_c = sin.contiguous().to(qkv.scalar_type());
  auto grad_q_c = grad_q.contiguous();
  auto grad_k_c = grad_k.contiguous();
  auto grad_v_c = grad_v.contiguous();

  const int batch_size = static_cast<int>(qkv_c.size(0));
  const int seq_len = static_cast<int>(qkv_c.size(1));
  const int q_dim = static_cast<int>(num_heads * head_dim);
  const int kv_dim = static_cast<int>(num_kv_heads * head_dim);
  const int qkv_dim = static_cast<int>(qkv_c.size(2));
  auto grad_qkv = torch::zeros_like(qkv_c);
  auto grad_q_gain = torch::zeros_like(q_gain_f);
  auto stream = at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      qkv_c.scalar_type(),
      "attention_prep_bshd_backward_cuda",
      [&] {
        launch_attention_prep_bshd_backward<scalar_t>(
            qkv_c,
            q_gain_f,
            cos_c,
            sin_c,
            grad_q_c,
            grad_k_c,
            grad_v_c,
            grad_qkv,
            grad_q_gain,
            batch_size,
            seq_len,
            static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            q_dim,
            kv_dim,
            qkv_dim,
            stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_qkv, grad_q_gain.to(q_gain.scalar_type())};
}
