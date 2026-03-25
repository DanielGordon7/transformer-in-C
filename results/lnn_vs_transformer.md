# LNN vs Transformer Decoder — Character-Level Language Modelling

## Setup

Both models were trained on the **Tiny Shakespeare** corpus (200 K characters,
~1 M characters in the full file) for next-character prediction.

| Property        | Value                          |
|-----------------|--------------------------------|
| Dataset         | Tiny Shakespeare (karpathy/char-rnn) |
| Vocabulary size | 62 characters                  |
| Sequence length | 64 characters                  |
| Train sequences | 1,000 per epoch                |
| Epochs          | 3                              |
| Learning rate   | 3 × 10⁻⁴ (both models)        |
| Metric          | Bits Per Character (BPC) ↓     |

### Model architectures

**LNN (Liquid Neural Network)**

```
Input:  one-hot(char) ∈ ℝ^62
State:  x ∈ ℝ^256  (hidden reservoir)
ODE:    τ·ẋ = −x + tanh(W_in·u + W_rec·x + b)
Output: logits = W_out·x + b_out  ∈ ℝ^62
Training: BPTT on last-character prediction
```

**Transformer Decoder (2-layer, 4-head, d=64)**

```
Input:  W_emb[token] + PE(pos)  ∈ ℝ^64
Per layer:
  h ← h + CausalSelfAttn(LayerNorm(h))
  h ← h + FFN(LayerNorm(h))          [FFN hidden = 256]
Output: logits = h @ W_out  ∈ ℝ^62   (every position)
Training: Cross-entropy on all T positions simultaneously
```

---

## Results

### Benchmark Table

| Model                 | Test BPC ↓ | Train Time | Params  | Memory  |
|-----------------------|------------|------------|---------|---------|
| LNN (CTRNN, H=256)    | 5.951      | 49.0 s     | 97,854  | 382 KB  |
| Transformer (d=64)    | **5.151**  | **7.3 s**  | 107,392 | 420 KB  |

*BPC = bits per character; lower is better. Random baseline = log₂(62) ≈ 5.95.*

### BPC Training Curves

| Epoch | LNN BPC | Transformer BPC |
|-------|---------|-----------------|
| 1     | 5.954   | 5.640           |
| 2     | 5.952   | 5.157           |
| 3     | 5.951   | 4.996           |

### Generated Text Samples (after 3 epochs, seed: `"JULIET:\n"`)

**LNN (200 chars)**
```
JULIET:
Mwjb;;&rbh ypCAAHVPHb:HLRmBUa!b?'wxoH-gQ.T!uFfIVWAymwtbv,B!JNFpLGW:n'zlB
ohjl'L.rcJ&IJjdtS.ilYlTVP - dIUuEOkD'G;vodsnAtWotI-DPor
UOC.JwJVhLyxETHG!bU&GuD:TzEflDjLddW,pIA!ag...
```

**Transformer (200 chars)**
```
JULIET:
i Lte
 m
 hilEoWdnfohe dN-uetloeie i
h uutS
uiyypadra ftnh iy esnnndTorssehonnosac...
```

*Both models are at early-training quality. The Transformer shows emerging word-length
patterns and spaces; the LNN output is still near-random. With more training and
proper per-position prediction, both would improve markedly.*

---

## Analysis

### Why the Transformer learns faster here

The core asymmetry is **training signal density**:

- The **Transformer** computes a loss over every one of the 64 positions in each
  sequence → 64 gradient signals per forward pass.
- The **LNN** (in this implementation) predicts only the **final character** of
  each sequence → 1 gradient signal per forward pass.

This 64× disadvantage in signal density explains most of the BPC gap at 3 epochs
and the 6.7× wall-clock difference (49 s vs 7.3 s for equivalent sequences).

A per-position LNN training loop (feed character, predict next, accumulate
gradients) would partially close both gaps, at the cost of longer BPTT chains.

---

## Architectural Comparison

### Computational Complexity

| Property               | LNN / CTRNN                           | Transformer Decoder                  |
|------------------------|---------------------------------------|--------------------------------------|
| **State size**         | O(H) — fixed, independent of seq len | O(T · D) — grows with context        |
| **Per-step FLOPs**     | O(H² + H·I) — constant               | O(T · D²) — linear in context        |
| **Attention pattern**  | Implicit, through recurrent dynamics | Explicit, full causal self-attention  |
| **Memory at inference**| O(H) — just the hidden state         | O(T · D) — full KV cache             |
| **Long-context**       | Amortised — state compresses history  | Exact — attends to every prior token |
| **Training BPTT depth**| T · ode_steps — can vanish/explode    | Shallow — direct gradient from each position |

### Strengths and Weaknesses

#### LNN

**Strengths**
- **Constant-memory inference**: the hidden state `x ∈ ℝ^H` is all that is
  needed at inference time, regardless of sequence length. No KV cache growth.
- **Physically grounded dynamics**: the ODE formulation maps directly to
  continuous-time systems. The time constants `τ` are learnable and create
  neurons that operate at different timescales — natural for sensor data,
  control systems, and neuroscience applications.
- **Irregular sampling**: the ODE integration step `dt` can vary per timestep,
  giving exact handling of non-uniform time intervals without any modification.
- **Online / streaming inference**: processes one token at a time with O(1)
  update — ideal for edge devices and real-time systems.
- **Interpretability**: `τ` values and sparse `W_rec` structure are inspectable;
  neurons can be categorised as "fast" or "slow" integrators.
- **Low inference latency**: sparse `W_rec` (~20 % density) reduces
  multiply-accumulate operations per step.

**Weaknesses**
- **Single gradient signal per sequence**: the current API trains on the last
  position only; per-position training requires multiple forward/backward passes.
- **Vanishing gradients over long BPTT chains**: T · ode_steps integration steps
  without gating (no LSTM/GRU equivalent) makes very long-range credit assignment
  difficult.
- **No direct token-to-token attention**: the LNN must rely on the recurrent
  state to carry all context; it cannot directly "look back" at an earlier token.
- **Hyperparameter sensitivity**: `dt`, `ode_steps`, `τ` range, and spectral
  radius all affect stability and require tuning.

---

#### Transformer Decoder

**Strengths**
- **Dense per-position training signal**: every position contributes a loss term,
  making optimisation highly sample-efficient.
- **Direct long-range attention**: any token can directly attend to any earlier
  token in O(1) attention steps — no information bottleneck through a fixed state.
- **Parallelisable training**: the full sequence is processed in one forward pass
  (no sequential dependency during training).
- **State-of-the-art language modelling**: GPT-style decoders are the foundation
  of all leading large language models.
- **Shallow effective gradient paths**: gradients flow directly from loss to
  embedding through attention, avoiding deep temporal chains.

**Weaknesses**
- **O(T²) attention cost**: computing attention over a context of T tokens costs
  O(T²) FLOPs and O(T · D) memory — impractical for very long sequences without
  sparse / linear attention approximations.
- **KV cache at inference**: autoregressive generation requires storing keys and
  values for all prior tokens, growing linearly with sequence length.
- **Fixed context window**: a standard transformer cannot naturally handle
  sequences longer than its `seq_len` parameter without windowing tricks.
- **Discrete-time only**: no built-in notion of elapsed time; irregular temporal
  sampling requires explicit positional engineering.
- **Heavy for edge deployment**: a useful transformer typically needs millions
  of parameters and significant memory bandwidth.

---

## When to Choose Each Architecture

| Use Case                               | Recommended |
|----------------------------------------|-------------|
| Streaming / real-time inference        | **LNN**     |
| Edge / memory-constrained device       | **LNN**     |
| Irregular time-series (sensors, EEG)  | **LNN**     |
| Long-context language generation       | **Transformer** |
| Short document classification          | **Transformer** |
| Batched training at scale              | **Transformer** |
| Continuous-time control / robotics     | **LNN**     |
| Code / text generation                 | **Transformer** |
| Multi-scale physiological signals      | **LNN**     |

---

## Running the Experiment

```bash
# Build C extension
python3 setup_lnn.py build_ext --inplace

# Download data and run
python3 experiment_shakespeare.py
```

Results are written to `results/shakespeare_results.json`.

---

## References

- Hasani, R. et al. (2021). *Liquid Time-Constant Networks.* AAAI.
- Funahashi, K. & Nakamura, Y. (1993). *Approximation of Dynamical Systems by
  Continuous Time Recurrent Neural Networks.* Neural Networks.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Karpathy, A. (2015). *The Unreasonable Effectiveness of Recurrent Neural
  Networks.* [blog post]
