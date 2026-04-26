# OLMo Hybrid 7B Reference

OLMo Hybrid is the main external architectural reference for the HGDN branch.

## 1. What It Is

OLMo Hybrid 7B is a **hybrid RNN-Transformer model** from AllenAI that replaces 75% of standard attention layers with Gated DeltaNet (GDN) recurrent layers. The core claim is ~2x data efficiency over OLMo 3 7B on core evals and 75% improvement in inference throughput/memory at long context lengths.

It is not a new architecture from scratch. It is a **surgical modification** of OLMo 3 7B, with three targeted changes:

1. Replace 3 out of every 4 attention layers with GDN recurrent layers
2. Reduce head count (`32 -> 30`) and `d_model` (`4096 -> 3840`) to equalize parameter count
3. Switch LR schedule from OLMo 3's piecewise schedule to standard cosine decay

---

## 2. Architecture Specification

| Property        | OLMo 3 7B     | **OLMo Hybrid 7B**            |
| --------------- | ------------- | ----------------------------- |
| Layers          | 32            | 32                            |
| d_model         | 4096          | **3840**                      |
| Q/KV Heads      | 32 / 32       | **30 / 30**                   |
| Head Dimension  | 128           | 128                           |
| GDN Heads       | -             | **30**                        |
| Layer Pattern   | all attention | **[GDN, GDN, GDN, Attn] x 8** |
| Context Length  | 65,536        | 65,536                        |
| Training Tokens | 5.93T         | 5.50T                         |

### Why Reduce Heads?

GDN layers add extra projections (`w_a`, `w_b`, `w_g`, short convolutions, `A_log`, `dt_bias`) that a vanilla attention layer does not have. To keep the total parameter count and tokens-per-second **comparable** to the OLMo 3 7B baseline for a fair comparison, 2 heads are removed. Each removed head shrinks d_model by one head dimension (128), giving d_model = 4096 - (2 x 128) = **3840**.

> **Note:** The codebase explicitly states that if training from scratch without a fair-comparison constraint, using n_heads=32 (a power of 2) is recommended.

---

## 3. The Gated DeltaNet Layer

GDN is a linear RNN sequence mixer based on the **chunk gated delta rule**. It replaces softmax attention's `O(n^2)` compute with `O(n)` recurrent state updates, at the cost of bounded state capacity.

### 3.1 Projections

Given input `x (B, T, d_model)`, the GDN layer computes:

```
q, k = w_q(x), w_k(x)          shape: (B, T, n_heads x head_k_dim)
v = w_v(x)                    shape: (B, T, n_v_heads x head_v_dim)
beta = sigmoid(w_b(x))        scalar gate (0, 1) per head
g_dt = -exp(A_log) * softplus(w_a(x) + dt_bias)   decay rate
```

The branch initializes `A_log` and `dt_bias` with the same timescale prior used
by public FLA: `A` is sampled in a positive range up to 16, and `dt_bias` is the
inverse-softplus of a log-uniform `dt` in `[0.001, 0.1]`. These parameters stay
in fp32 and are optimized in a no-weight-decay Adam group. Zero initialization
would start around `exp(-softplus(0)) ~= 0.5` retention per token, which is not
an OLMo/FLA-faithful GDN starting point.

### 3.2 Head Dimensions

OLMo Hybrid uses **asymmetric head dimensions**:

| Dimension          | Formula                      | Value             |
| ------------------ | ---------------------------- | ----------------- |
| Key/Query head dim | `0.75 x (d_model / n_heads)` | `0.75 x 128 = 96` |
| Value head dim     | `key_head_dim x expand_v`    | `96 x 2.0 = 192`  |

The value head dimension being wider than the key/query dimension directly **increases the fixed-size recurrent state** `(n_v_heads, head_k_dim, head_v_dim)`, improving the model's capacity to compress long-range context without any memory scaling cost (unlike the KV cache in softmax attention, which grows with sequence length).

### 3.3 Negative Eigenvalues (`allow_neg_eigval=True`)

When enabled, `beta` is multiplied by 2.0:

```python
beta = sigmoid(w_b(x)) * 2.0   # range: (0, 2) instead of (0, 1)
```

This allows the recurrent state update to use eigenvalue magnitudes > 1 transiently, enabling the model to track patterns that require **sign alternation** in the state. This comes from the paper [*Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*](https://arxiv.org/abs/2411.12537) and is a meaningful capability extension - standard linear RNNs with beta (0,1) cannot represent oscillatory or sign-alternating dynamics.

### 3.4 Short Convolutions

Before the recurrent kernel, q, k, v each pass through a **causal conv1d** with kernel size 4 and SiLU activation. This provides local context aggregation that the purely recurrent kernel cannot capture efficiently, a design pattern common across Mamba, Hawk, and similar hybrid RNNs.

### 3.5 Recurrent Kernel & Output

```python
o, _ = dispatch_chunk_gated_delta_rule(q, k, v, g, beta, ...)
# Output gating
g_out = w_g(x).view(B, T, n_v_heads, head_v_dim)
out   = w_out(FusedRMSNormGated(o, g_out).view(B, T, -1))
```

The chunked kernel processes sequences in blocks for hardware efficiency. The output is gated with a separate `w_g` projection before the final linear projection back to d_model.
The custom HGDN path keeps the norm/gate in PyTorch or the optional sidecar
output op, but includes the learned per-`head_v_dim` norm weight so the
parameterization matches the important part of `FusedRMSNormGated`.

### 3.6 GDN Parameter Count (per layer)

```
w_q, w_k:        2 x d_model x (n_heads x head_k_dim)
w_v:             d_model x (n_v_heads x head_v_dim)
w_a, w_b:        2 x d_model x n_v_heads
w_g:             d_model x (n_v_heads x head_v_dim)
w_out:           (n_v_heads x head_v_dim) x d_model
A_log, dt_bias:  2 x n_v_heads
q_conv, k_conv:  2 x conv_size x (n_heads x head_k_dim)
v_conv:          conv_size x (n_v_heads x head_v_dim)
o_norm:          head_v_dim
```

---

## 4. Layer Interleaving Pattern

```
Layer index:   0    1    2    3    4    5    6    7   ...  28   29   30   31
Layer type:   GDN  GDN  GDN  ATN  GDN  GDN  GDN  ATN ...  GDN  GDN  GDN  ATN
```

The pattern `["gdn", "gdn", "gdn", "attn"]` repeats 8 times across 32 layers, yielding **24 GDN + 8 attention** layers. The attention layers are standard OLMo 3-style multi-head attention with:

- `qk_norm=True`
- `rope_theta=500,000`
- Sliding window pattern `[4096, 4096, 4096, -1]` (every 4th layer is full attention)

---

## 5. Training Pipeline

### Stage 1 - Pretraining

| Property          | Value                                                         |
| ----------------- | ------------------------------------------------------------- |
| Data              | OLMo-mix-0925 (dolma3)                                        |
| Sequence length   | 8,192                                                         |
| Global batch size | ~4M tokens                                                    |
| Optimizer         | SkipStepAdamW (`lr=3e-4`, `wd=0.1`, `betas=(0.9, 0.95)`)     |
| **LR Schedule**   | **CosWithWarmup (warmup=2000 steps)** key change vs OLMo 3 7B |
| Parallelism       | HSDP, bf16 params / fp32 reduce                               |
| RoPE              | Enabled                                                       |

### Stage 2 - Midtraining

| Property | Value                                                                    |
| -------- | ------------------------------------------------------------------------ |
| Data     | OLMo3-32B data mix (improved vs OLMo3 7B mix)                            |
| Schedule | LinearWithWarmup (decay to 0)                                            |
| Soups    | Two independent runs (seeds 1337, 683) averaged into a single checkpoint |

### Stage 3 - Long-Context Extension

| Property            | Value                                                                |
| ------------------- | -------------------------------------------------------------------- |
| Sequence length     | 65,536                                                               |
| Data                | dolma3-longmino                                                      |
| **RoPE**            | **Dropped (DroPE)** - removed from all attention layers              |
| Context parallelism | Ulysses, degree=2                                                    |
| Memory              | Fused linear loss; activation checkpointing (budget mode, 10%)       |
| Hardware note       | 30 heads constrains Ulysses to degree=2 only; requires B200s for HBM |

> **Why DroPE?** Sinusoidal positional encodings (RoPE) degrade at sequence lengths far beyond their training range. Dropping RoPE entirely at the long-context stage lets positional information emerge from the attention mask and recurrent state dynamics rather than explicit position embeddings.

---

## 6. Architecture Construction (Code Summary)

```python
# 1. Start from OLMo 3 7B base
model_config = TransformerConfig.olmo3_7B(vocab_size=vocab_size)

# 2. Reduce heads and d_model to equalize params vs. pure-transformer baseline
REMOVE_HEADS = 2
model_config.d_model -= REMOVE_HEADS * 128          # 4096 -> 3840
num_heads = attn_cfg.n_heads - REMOVE_HEADS         # 32 -> 30
# head_dim stays fixed at 128

# 3. Build GDN block
gdn_block = attn_block.replace(
    sequence_mixer=GatedDeltaNetConfig(
        n_heads=30,
        head_dim=int(0.75 * 3840 / 30),   # = 96  (key/query dim)
        expand_v=2.0,                       # val dim = 192
        allow_neg_eigval=True,              # beta range (0, 2)
        conv_size=4,
    )
)

# 4. Set interleaved block pattern
model_config.block = {"gdn": gdn_block, "attn": attn_block}
model_config.block_pattern = ["gdn", "gdn", "gdn", "attn"]  # x 8 = 32 layers
```

---

## 7. Key Design Decisions & Tradeoffs

| Decision                    | Rationale                                                                                | Tradeoff                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 3:1 GDN:Attn ratio          | Maximize recurrent efficiency while retaining periodic full attention for global context | Higher ratio would risk longer-range forgetting            |
| expand_v=2.0                | Wider value heads larger fixed recurrent state better compression                        | More parameters per GDN layer                              |
| allow_neg_eigval            | Enables oscillatory/sign-alternating state tracking                                      | Slightly less stable training (beta can exceed 1)          |
| DroPE at long context       | Avoids RoPE degradation at 65k+ lengths                                                  | Positional information becomes fully implicit              |
| Cosine vs. piecewise LR     | Cleaner schedule, standard in hybrid RNN literature                                      | Loses OLMo 3's carefully tuned piecewise decay stages      |
| Model soups at midtraining  | Averaging two seeds improves generalization with no extra inference cost                 | Requires 2x compute for midtraining stage                  |
| n_heads=30 (non-power-of-2) | Required by param-equalization constraint                                                | Limits Ulysses CP to degree=2; constrains hardware choices |

---

## 8. Dependencies

- **`flash-linear-attention` (`fla`)**: Required for `dispatch_chunk_gated_delta_rule` kernel. GDN instantiation raises at runtime if `has_fla()` returns False.
- **`FusedRMSNormGated`**: From `fla.modules`, used as the output norm within GDN.
- Parallelism: HSDP (pretraining) FSDP (long-context); Ulysses context parallelism (long-context only).

---

## 9. Formal GDN Recurrence

The paper defines the GDN state update precisely. For each token $t$, given $q_t, k_t \in \mathbb{R}^d$, $v_t \in \mathbb{R}^{2d}$, and scalars $\alpha_t, \beta_t \in (0,1)$ with $\|k_t\| = 1$:

$$S_t = S_{t-1} \alpha_t (I - 2\beta_t k_t k_t^\top) + v_t k_t^\top$$

$$y_t = S_t q_t$$

The key term is $(I - 2\beta_t k_t k_t^\top)$: this is a Householder-like reflection applied to the previous state $S_{t-1}$, scaled by the decay $\alpha_t$. Without negative eigenvalues ($\beta_t \in (0, 0.5)$), the reflection has eigenvalues $\{1, 1 - 2\beta_t\}$, both positive. With negative eigenvalues ($\beta_t \in (0, 1)$ after the $\times 2$ trick), the second eigenvalue can go negative, enabling the state matrix to implement a **swap operator** - the key to expressing state-tracking tasks like parity. This extends original DeltaNet (Schlag et al., 2021) by adding the per-token decay factor $\alpha_t$.

---

## 10. Inference State Size

One of the paper's most concrete practical results is the inference memory comparison. GDN's recurrent state is constant-size regardless of sequence length, unlike KV caches which scale with context:

| Layer Type                     | Elements  | FP16 Size    | vs. GDN   |
| ------------------------------ | --------- | ------------ | --------- |
| MHA (32K seq, 32 KV heads)     | 268.4M    | 512 MiB      | **485x**  |
| GQA (32K seq, 8 KV heads)      | 67.1M     | 128 MiB      | **121x**  |
| GQA SWA (4096 window)          | 8.39M     | 16.0 MiB     | **15.2x** |
| **OLMo Hybrid GDN** (30 heads) | **0.55M** | **1.05 MiB** | -         |

The GDN state is only ~1 MB per layer. This is what drives the long-context inference efficiency gains - not faster computation per token, but dramatically less memory pressure, enabling larger batches and more concurrency at long sequences.

---

## 11. Expressivity Theory

The paper makes a formal theoretical argument for *why* hybrid models should be better, rooted in computational complexity.

### 11.1 The Expressivity-Parallelism Frontier

The core thesis is that a good architecture should maximize expressivity while preserving parallelizability. The complexity hierarchy relevant here:

```
TC0 < NC1 < PNC1
```

- **Transformers** are bounded by TC0. They cannot robustly solve state-tracking tasks (e.g., the shell game, composing permutations) under the standard conjecture TC0 = NC1, because attention cannot aggregate sequential state updates in an order-sensitive way across arbitrary lengths.
- **GDN (linear RNNs)** can express NC1-complete problems including state tracking, by using time-dependent non-diagonal transition matrices. The negative eigenvalue extension is critical here - without it, GDN is still limited to TC0.
- **Linear RNNs are bounded by recall capacity.** Their fixed-size hidden state limits performance on copy/retrieval tasks. The state must compress an entire context prefix into a constant-size matrix, which fails for tasks requiring `O(n)` bits from prefix to suffix.

### 11.2 Hybrid Models Are More Than the Sum of Their Parts (Theorem 1)

The paper proves that hybrids can solve **state-based recall** - a task neither transformers nor GDN alone can express:

**State-Based Recall:** Given a bitstring $x$, pointers $p_1,…,p_5$, and a sequence of transpositions $\pi$, compute $x_{q_1}$ where $q = (\pi_n \circ … \circ \pi_1)(p)$.

This requires *both* composing state updates (transpositions over pointers) AND recalling a value from a long context - beyond either primitive alone.

**Theorem 1:** A hybrid model with a single alternation (GDN then attention, or attention then GDN) can solve state-based recall. No transformer or pure GDN can (under TC0 = NC1).

A concrete code analog from the paper:

```python
# State tracking: composing swaps over variables
a, b, c, d, e = range(5)
a, c = c, e   # ... n lines of swaps
# Recall: indexing a long array with the tracked variable
bits = [0, 1, 0, 0, ...]  # m bits
assert bits[a] == _
```

Transformers fail as $n$ grows (can't track state). Linear RNNs fail as $m$ grows (can't recall from large arrays). Hybrids handle both.

### 11.3 Padded Hybrid Models Cover All of NC1 (Theorem 3)

With polynomial padding tokens, hybrid models (averaging-hard attention + GDN with negative eigenvalues) can recognize **any language in NC1**, while padded transformers are bounded at TC0. Since NC1 is conjectured strictly larger than TC0, this is a meaningful expressivity gap. A corollary: hybrid models can evaluate arbitrary boolean formulas; neither transformers nor pure linear RNNs can.

### 11.4 Empirical Validation on Synthetic Tasks

The paper trains small transformer, linear RNN, and hybrid models on three synthetic tasks:

| Task                          | Transformer                | Linear RNN          | Hybrid                      |
| ----------------------------- | -------------------------- | ------------------- | --------------------------- |
| State tracking (n swaps)      | Fails rapidly (22% at n=8) | Near-perfect        | Near-perfect                |
| Recall (m-bit array)          | Perfect                    | Degrades at large m | Perfect                     |
| State-based recall (combined) | Degrades                   | Degrades            | Near-perfect across all n,m |

Results match theory exactly.

---

## 12. Scaling Laws

### 12.1 Chinchilla-Style Fit

The paper fits $L(N,D) = E + A/N^\alpha + B/D^\beta$ to 60M-1B parameter models across 5 Chinchilla multiples per size, using a WSD-S (warmup-stable-decay with periodic resets) schedule to collect multiple data points from a single run.

**Fixed-exponent fit** ( = = 0.22 shared, only E, A, B refit - the statistically robust comparison):

| Architecture   | E    | A    | **B**                  |
| -------------- | ---- | ---- | ---------------------- |
| Transformer    | 1.55 | 66.6 | **94.9** [88.7, 102.0] |
| **Hybrid GDN** | 1.58 | 65.1 | **83.7** [80.2, 87.1]  |
| Pure GDN       | 1.61 | 62.4 | 90.8 [87.4, 93.3]      |

The **data efficiency coefficient B** is the only parameter with non-overlapping 95% CIs between hybrid and transformer. The scaling exponents and are statistically indistinguishable - consistent with theory, which predicts expressivity shifts the efficiency *constants* without changing the *exponents*.

The prediction on the released models validates the fit: predicted loss for OLMo Hybrid 7B (5.5T tokens) was 2.142, observed was 2.136 (0.28% error).

### 12.2 Projected Token Savings

At matched model size, the hybrid requires fewer tokens to reach the same loss. Token savings grow with scale:

| Model Size | Token Savings |
| ---------- | ------------- |
| 1B         | 1.32x         |
| 7B         | 1.68x         |
| 30B        | 1.82x         |
| 70B        | 1.89x         |

The pretraining run confirms this: OLMo Hybrid reaches the same MMLU accuracy as OLMo 3 7B using **49% fewer tokens**, and the same Common Crawl CE loss using **35% fewer tokens**.

### 12.3 Theoretical Explanation: Quantization Model

The paper adapts Michaud et al.'s (2023) quantization model to explain *why* expressivity improves scaling. The key assumptions: language modeling is a multi-task problem where tasks are either expressible or not by a given architecture. Inexpressible tasks require more parameters (C' > C) and/or more tokens (T' > T) to approximate, and achieve lower loss reduction (' ) when learned.

**Theorem 4** (paper): Under these assumptions, the data efficiency coefficient is:

$$B_\epsilon = (1-\epsilon)\Delta T^{\alpha/(\alpha+1)} + \epsilon\Delta' T'^{\alpha/(\alpha+1)}$$

Decreasing $\epsilon$ (more expressive architecture) strictly decreases $B_\epsilon$ fewer tokens needed for the same loss. Scaling *exponents* are independent of $\epsilon$. This matches observations: the hybrid's advantage is in B (data efficiency), not in or .

---

## 13. Architecture Ablation Results

The paper ran controlled ablations at 60M-1B scale to justify the final design choices:

### RNN Backbone: GDN vs. Mamba2

Mamba2-based hybrids underperformed the transformer baseline at every scale (0.879 vs. 0.747 BPB at 760M). Pure GDN outperformed the transformer at all scales. GDN is the correct choice; the gap does not close with scale.

### Layer Placement: Interleaved vs. Middle

Interleaving attention every 4th layer consistently beats concentrating attention layers in the middle of the network. The paper's interpretation: uniform distribution lets every part of the network access global context, and maximizes layer-type alternations which may unlock additional expressivity (per Theorem 1, each alternation can unlock more capability).

### GDN:Attention Ratio

- **1:1 (50% attention):** Competitive at small scales, underperforms at large scales - extra attention cost doesn't pay off.
- **3:1 (25% attention):** Best or second-best across all large scales and domains. Selected for OLMo Hybrid.
- **7:1 (12.5% attention):** Competitive at small scales, slightly degrades at large scales despite more parameters (more GDN layers).

The takeaway: 3:1 is a robust default, but the exact ratio matters less than the decision to hybridize at all - all three interleaved configs substantially beat the transformer at large scales.

---

## 14. Training Stability

The paper quantifies training stability via **spike score**: percentage of gradient norm values 6 standard deviations from a rolling 128-step average.

OLMo 3 shows a growing spike score throughout training. OLMo Hybrid shows a flat, low trajectory. The authors interpret this as the hybrid being more tolerant of large learning rates and noisy data - a secondary benefit beyond expressivity that may also contribute to its scaling advantage.

---

## 15. Post-Training Observations and Open Problems

The paper is candid about post-training being harder for hybrid models:

**Positive:** Stronger pretraining translates to persistent gains on knowledge tasks (Think SFT: +5.6% MMLU, +4.3% PopQA over OLMo 3).

**Negative/Open:**

- Extended reasoning tasks (AIME, Omega) lag behind OLMo 3 - the hybrid has not yet benefited from post-training data optimized for its architecture.
- Inference requires `--enforce-eager` (disables torch compilation) for numerical correctness, because torch compilation introduces subtle numerical differences that compound across recurrent GDN state updates. This substantially reduces throughput (roughly 0.5-0.9x that of dense baseline at long contexts). Workaround: storing the GDN cache in FP32 (`--mamba_ssm_cache_dtype float32`) and re-enabling compilation recovers similar scores.
- Ulysses context parallelism is constrained to degree=2 due to n_heads=30 being non-power-of-2, requiring B200s (not H100s) for the long-context training stage.
- Post-training for hybrid models is, in the authors' words, "in its infancy."

---

## 16. Related Work

| Paper                                                                                              | Relevance                                                                                                |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Gated Delta Networks (2412.06464)](https://arxiv.org/abs/2412.06464)                              | Defines the GDN layer used as the RNN mixer                                                              |
| [Unlocking State-Tracking via Negative Eigenvalues (2411.12537)](https://arxiv.org/abs/2411.12537) | Justifies `allow_neg_eigval=True` / betax2 trick                                                         |
| [DroPE (2512.12167)](https://arxiv.org/abs/2512.12167)                                             | Dropping RoPE for long-context extension                                                                 |
| Michaud et al., 2023 - Quantization Model                                                          | Theoretical framework for why expressivity improves scaling                                              |
| Merrill & Sabharwal, 2023 - Parallelism Tradeoff                                                   | Formalizes the TC0 upper bound on transformers                                                           |
| Grazzi et al., 2025 - Negative Eigenvalues in Linear RNNs                                          | Proves GDN with negative eigenvalues reaches NC1                                                         |
| Samba / Nemotron-H / Falcon H1 / Kimi Linear                                                       | Concurrent large-scale hybrid releases; OLMo Hybrid is on Pareto frontier among open-weight dense models |
| Mamba / Mamba-2                                                                                    | Ablated and found inferior to GDN at all tested scales                                                   |
| Griffin / Hawk                                                                                     | Precedent for interleaved attention + linear RNN layer patterns                                          |
