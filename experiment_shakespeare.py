"""
Shakespeare Experiment: LNN vs Transformer Decoder
Both models trained on character-level next-token prediction (Tiny Shakespeare).
Metrics: bits-per-character (BPC), training time, parameters, sample generation.
"""

import sys, os, time, math, random, urllib.request
import numpy as np
import lnn as lnn_module

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Data ────────────────────────────────────────────────────────────────────

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
SHAKESPEARE_PATH = "shakespeare.txt"

def download_shakespeare():
    if os.path.exists(SHAKESPEARE_PATH):
        return True
    print("Downloading Tiny Shakespeare …")
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
        print("  Done.")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def load_data(path, max_chars=200_000):
    with open(path, encoding="utf-8") as f:
        text = f.read()[:max_chars]
    chars  = sorted(set(text))
    vocab  = {c: i for i, c in enumerate(chars)}
    rvocab = {i: c for c, i in vocab.items()}
    tokens = np.array([vocab[c] for c in text], dtype=np.int32)
    return tokens, vocab, rvocab, len(chars)

SEQ_LEN   = 64     # characters per training sequence
BATCH_SKIP = 8     # stride between sequences

def make_sequences(tokens, seq_len=SEQ_LEN, skip=BATCH_SKIP):
    seqs = []
    for i in range(0, len(tokens) - seq_len - 1, skip):
        seqs.append(tokens[i : i + seq_len + 1])   # seq_len input + 1 target
    return np.array(seqs, dtype=np.int32)

# ── Transformer decoder (numpy, 2-layer, causal) ───────────────────────────

class TransformerDecoder:
    """
    Minimal 2-layer causal transformer decoder in pure numpy.
    Architecture per layer:
        h = h + CausalSelfAttn(LayerNorm(h))
        h = h + FFN(LayerNorm(h))
    Output: h @ W_out  (T, vocab_size)
    Training: cross-entropy loss, SGD with gradient clipping.
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 seq_len=SEQ_LEN, seed=SEED):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.seq_len    = seq_len
        self.d_k        = d_model // n_heads

        def xavier(rows, cols):
            s = math.sqrt(6.0 / (rows + cols))
            return rng.uniform(-s, s, (rows, cols)).astype(np.float32)

        # Token embedding  (vocab_size, d_model)
        self.W_emb = xavier(vocab_size, d_model)

        # Positional encoding (fixed, sinusoidal)
        self.PE = self._make_pe(seq_len, d_model)

        # Per-layer weights
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Attention
                "W_Q":  xavier(d_model, d_model),
                "W_K":  xavier(d_model, d_model),
                "W_V":  xavier(d_model, d_model),
                "W_O":  xavier(d_model, d_model),
                # LayerNorm 1 (before attn)
                "gamma1": np.ones(d_model,  dtype=np.float32),
                "beta1":  np.zeros(d_model, dtype=np.float32),
                # FFN
                "W1":  xavier(d_model, d_model * 4),
                "b1":  np.zeros(d_model * 4, dtype=np.float32),
                "W2":  xavier(d_model * 4, d_model),
                "b2":  np.zeros(d_model, dtype=np.float32),
                # LayerNorm 2 (before FFN)
                "gamma2": np.ones(d_model,  dtype=np.float32),
                "beta2":  np.zeros(d_model, dtype=np.float32),
            }
            self.layers.append(layer)

        # Output projection
        self.W_out = xavier(d_model, vocab_size)

    @staticmethod
    def _make_pe(seq_len, d_model):
        pe = np.zeros((seq_len, d_model), dtype=np.float32)
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
        return pe

    @property
    def n_params(self):
        total = self.W_emb.size + self.W_out.size
        for l in self.layers:
            total += sum(v.size for v in l.values())
        return total

    # ── Layer-norm helpers ────────────────────────────────────────────────

    @staticmethod
    def _ln_forward(x, gamma, beta, eps=1e-5):
        mean = x.mean(-1, keepdims=True)
        var  = x.var(-1,  keepdims=True)
        xn   = (x - mean) / np.sqrt(var + eps)
        return gamma * xn + beta, xn, var, mean

    @staticmethod
    def _ln_backward(dout, xn, var, gamma, eps=1e-5):
        T, D = dout.shape
        dgamma = (dout * xn).sum(0)
        dbeta  = dout.sum(0)
        dxn    = dout * gamma
        dvar   = (-0.5 * dxn * xn / (var + eps)).sum(-1, keepdims=True)
        dmean  = (-dxn / np.sqrt(var + eps)).sum(-1, keepdims=True)
        dx     = (dxn / np.sqrt(var + eps)
                  + 2.0 * dvar * xn / D
                  + dmean / D)
        return dx, dgamma, dbeta

    # ── Single-layer forward ──────────────────────────────────────────────

    def _layer_forward(self, x, layer, cache):
        T, D = x.shape
        H = self.n_heads; dk = self.d_k

        # --- Attn sub-layer ---
        xn1, xn1_norm, var1, mean1 = self._ln_forward(
            x, layer["gamma1"], layer["beta1"])

        Q = xn1 @ layer["W_Q"]   # (T, D)
        K = xn1 @ layer["W_K"]
        V = xn1 @ layer["W_V"]

        # Multi-head reshape  (T, H, dk)
        Qh = Q.reshape(T, H, dk).transpose(1, 0, 2)   # (H, T, dk)
        Kh = K.reshape(T, H, dk).transpose(1, 0, 2)
        Vh = V.reshape(T, H, dk).transpose(1, 0, 2)

        scale = math.sqrt(dk)
        S = (Qh @ Kh.transpose(0, 2, 1)) / scale      # (H, T, T)

        # Causal mask
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), 1)
        S += mask

        # Softmax
        S_max = S.max(-1, keepdims=True)
        A_exp = np.exp(S - S_max)
        A = A_exp / A_exp.sum(-1, keepdims=True)       # (H, T, T)

        O_h = A @ Vh                                   # (H, T, dk)
        O   = O_h.transpose(1, 0, 2).reshape(T, D)    # (T, D)
        O_proj = O @ layer["W_O"]
        attn_out = x + O_proj

        cache.update(dict(xn1=xn1, xn1_norm=xn1_norm, var1=var1,
                          Q=Q, K=K, V=V, Qh=Qh, Kh=Kh, Vh=Vh,
                          A=A, O=O, O_proj=O_proj, attn_out=attn_out))

        # --- FFN sub-layer ---
        xn2, xn2_norm, var2, mean2 = self._ln_forward(
            attn_out, layer["gamma2"], layer["beta2"])

        z1    = xn2 @ layer["W1"] + layer["b1"]   # (T, 4D)
        relu  = np.maximum(0, z1)
        z2    = relu @ layer["W2"] + layer["b2"]  # (T, D)
        out   = attn_out + z2

        cache.update(dict(xn2=xn2, xn2_norm=xn2_norm, var2=var2,
                          z1=z1, relu=relu, ffn_out=out))
        return out

    # ── Single-layer backward ─────────────────────────────────────────────

    def _layer_backward(self, dout, x, layer, cache):
        T, D = dout.shape
        H = self.n_heads; dk = self.d_k
        grads = {}

        # --- FFN backward ---
        dz2 = dout.copy()
        grads["W2"] = cache["relu"].T @ dz2
        grads["b2"] = dz2.sum(0)
        d_relu = dz2 @ layer["W2"].T
        d_relu[cache["z1"] <= 0] = 0.0
        grads["W1"] = cache["xn2"].T @ d_relu
        grads["b1"] = d_relu.sum(0)
        dxn2 = d_relu @ layer["W1"].T

        d_attn_out, grads["gamma2"], grads["beta2"] = self._ln_backward(
            dxn2, cache["xn2_norm"], cache["var2"], layer["gamma2"])
        d_attn_out += dout   # residual

        # --- Attn backward ---
        grads["W_O"] = cache["O"].T @ d_attn_out
        dO = d_attn_out @ layer["W_O"].T

        # dO → multi-head
        dO_h = dO.reshape(T, H, dk).transpose(1, 0, 2)   # (H, T, dk)

        # dA = dO_h @ Vh^T
        dA  = dO_h @ cache["Vh"].transpose(0, 2, 1)       # (H, T, T)
        dVh = cache["A"].transpose(0, 2, 1) @ dO_h        # (H, T, dk)

        # Softmax backward
        dS  = cache["A"] * (dA - (dA * cache["A"]).sum(-1, keepdims=True))
        dS /= math.sqrt(dk)

        dQh = dS @ cache["Kh"]                             # (H, T, dk)
        dKh = dS.transpose(0, 2, 1) @ cache["Qh"]

        dQ = dQh.transpose(1, 0, 2).reshape(T, D)
        dK = dKh.transpose(1, 0, 2).reshape(T, D)
        dV = dVh.transpose(1, 0, 2).reshape(T, D)

        grads["W_Q"] = cache["xn1"].T @ dQ
        grads["W_K"] = cache["xn1"].T @ dK
        grads["W_V"] = cache["xn1"].T @ dV
        dxn1 = dQ @ layer["W_Q"].T + dK @ layer["W_K"].T + dV @ layer["W_V"].T

        dx, grads["gamma1"], grads["beta1"] = self._ln_backward(
            dxn1, cache["xn1_norm"], cache["var1"], layer["gamma1"])
        dx += d_attn_out   # residual

        return dx, grads

    # ── Full forward pass ────────────────────────────────────────────────

    def forward(self, token_ids):
        """token_ids: (T,) int → logits: (T, vocab_size)"""
        T = len(token_ids)
        h = self.W_emb[token_ids] + self.PE[:T]   # (T, d_model)
        self._h0    = h.copy()
        self._caches = []
        for layer in self.layers:
            cache = {}
            h = self._layer_forward(h, layer, cache)
            self._caches.append(cache)
        self._h_final = h
        return h @ self.W_out   # (T, vocab_size)

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(self, token_ids, lr=3e-4):
        """
        token_ids: (T+1,) — first T are input, last T are targets.
        Returns (loss, bpc).
        """
        T   = len(token_ids) - 1
        inp = token_ids[:-1]
        tgt = token_ids[1:]

        logits = self.forward(inp)   # (T, V)

        # Softmax + cross-entropy
        log_max = logits.max(-1, keepdims=True)
        exp_l   = np.exp(logits - log_max)
        probs   = exp_l / exp_l.sum(-1, keepdims=True)

        loss = -np.log(probs[np.arange(T), tgt] + 1e-9).mean()
        bpc  = loss / math.log(2)

        # Upstream gradient for softmax + CE
        d_logits = probs.copy()
        d_logits[np.arange(T), tgt] -= 1.0
        d_logits /= T

        # Output layer gradient
        dW_out = self._h_final.T @ d_logits   # (D, V)
        dh     = d_logits @ self.W_out.T      # (T, D)

        # Backprop through transformer layers (reverse order)
        layer_grads = [None] * self.n_layers
        for li in reversed(range(self.n_layers)):
            x_in = (self._h0 if li == 0
                    else self._caches[li - 1]["ffn_out"])
            dh, grads = self._layer_backward(dh, x_in, self.layers[li],
                                             self._caches[li])
            layer_grads[li] = grads

        # Embedding gradient
        dW_emb = np.zeros_like(self.W_emb)
        np.add.at(dW_emb, inp, dh)

        # Gradient clipping + update
        def clip_apply(param, grad):
            n = np.linalg.norm(grad)
            if n > 1.0: grad = grad * (1.0 / n)
            param -= lr * grad

        clip_apply(self.W_emb, dW_emb)
        clip_apply(self.W_out, dW_out)
        for li, grads in enumerate(layer_grads):
            for k, g in grads.items():
                clip_apply(self.layers[li][k], g)

        return float(loss), float(bpc)

    # ── Text generation ───────────────────────────────────────────────────

    def generate(self, seed_ids, n_chars, temperature=0.8):
        ctx = list(seed_ids[-self.seq_len:])
        out = []
        for _ in range(n_chars):
            ids = np.array(ctx[-self.seq_len:], dtype=np.int32)
            logits = self.forward(ids)[-1]   # last position
            logits = logits / temperature
            exp_l  = np.exp(logits - logits.max())
            probs  = exp_l / exp_l.sum()
            c = int(np.random.choice(len(probs), p=probs))
            out.append(c)
            ctx.append(c)
        return out


# ── LNN character-level model ──────────────────────────────────────────────

class LNNCharModel:
    def __init__(self, vocab_size, hidden_size=256, seq_len=SEQ_LEN, seed=SEED):
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.seq_len     = seq_len
        # input: one-hot of vocab_size; output: logits over vocab_size
        self.net = lnn_module.LiquidNN(
            vocab_size, hidden_size, vocab_size, dt=0.05, ode_steps=1)

    @property
    def n_params(self):
        R, I, O = (self.net.reservoir_size, self.net.input_size,
                   self.net.output_size)
        return R * I + R * R + O * R + R + O + R

    def _one_hot_seq(self, token_ids):
        """(T,) int → flat list of length T*vocab_size"""
        T  = len(token_ids)
        V  = self.vocab_size
        oh = np.zeros((T, V), dtype=np.float32)
        oh[np.arange(T), token_ids] = 1.0
        return oh.flatten().tolist()

    def train_step(self, token_ids, lr=3e-4):
        """token_ids: (T+1,)"""
        T   = len(token_ids) - 1
        inp = token_ids[:-1]   # (T,)
        tgt = token_ids[1:]    # (T,) — we train on *last* token target

        flat   = self._one_hot_seq(inp)
        target = np.zeros(self.vocab_size, dtype=np.float32)
        target[tgt[-1]] = 1.0   # predict final character

        self.net.reset_state()
        mse_loss = self.net.train_sequence(flat, T, target.tolist(), lr=lr)

        # Approximate BPC from MSE (not exact but comparable)
        # Run forward to get logit and compute proper CE
        self.net.reset_state()
        logits_raw = self.net.forward_sequence(flat, T)
        logits = np.array(logits_raw, dtype=np.float32)
        exp_l  = np.exp(logits - logits.max())
        probs  = exp_l / exp_l.sum()
        ce     = -math.log(float(probs[tgt[-1]]) + 1e-9)
        bpc    = ce / math.log(2)
        return mse_loss, bpc

    def generate(self, seed_ids, n_chars, temperature=0.8):
        self.net.reset_state()
        # Warm up on seed
        if len(seed_ids) > 1:
            flat = self._one_hot_seq(seed_ids[:-1])
            self.net.forward_sequence(flat, len(seed_ids) - 1)

        out = []
        current = seed_ids[-1]
        for _ in range(n_chars):
            oh = [0.0] * self.vocab_size
            oh[current] = 1.0
            logits_raw = self.net.forward(oh)
            logits = np.array(logits_raw, dtype=np.float32) / temperature
            exp_l  = np.exp(logits - logits.max())
            probs  = exp_l / exp_l.sum()
            c = int(np.random.choice(self.vocab_size, p=probs))
            out.append(c)
            current = c
        return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not download_shakespeare():
        sys.exit(1)

    print("Loading Tiny Shakespeare (200K chars) …")
    tokens, vocab, rvocab, vocab_size = load_data(SHAKESPEARE_PATH)
    seqs = make_sequences(tokens)
    print(f"  vocab={vocab_size}  sequences={len(seqs)}  seq_len={SEQ_LEN}")

    # Split 90/10
    n_train = int(0.9 * len(seqs))
    train_seqs = seqs[:n_train]
    test_seqs  = seqs[n_train:]

    EPOCHS    = 3
    LR        = 3e-4
    N_TRAIN   = 1000   # sequences per epoch
    N_TEST    = 200

    print(f"  training on {N_TRAIN} seqs/epoch, {EPOCHS} epochs\n")

    # ── LNN ──
    print("── LNN (reservoir=256) ──")
    lnn_model = LNNCharModel(vocab_size, hidden_size=256)
    idx = list(range(len(train_seqs)))

    t0 = time.time()
    lnn_bpc_curve = []
    for epoch in range(EPOCHS):
        random.shuffle(idx)
        ep_bpc = 0.0
        for i in idx[:N_TRAIN]:
            _, bpc = lnn_model.train_step(train_seqs[i], lr=LR)
            ep_bpc += bpc
        avg = ep_bpc / N_TRAIN
        lnn_bpc_curve.append(avg)
        print(f"  LNN epoch {epoch+1}/{EPOCHS}  train BPC={avg:.3f}")
    lnn_train_t = time.time() - t0

    # Eval on test
    lnn_test_bpc = 0.0
    for i in range(N_TEST):
        _, bpc = lnn_model.train_step(test_seqs[i % len(test_seqs)], lr=0.0)
        lnn_test_bpc += bpc
    lnn_test_bpc /= N_TEST

    # Generate sample
    seed_text = "JULIET:\n"
    seed_ids  = np.array([vocab.get(c, 0) for c in seed_text], dtype=np.int32)
    gen_ids   = lnn_model.generate(seed_ids, 200)
    lnn_sample = seed_text + "".join(rvocab[i] for i in gen_ids)

    # ── Transformer ──
    print("\n── Transformer (d_model=64, 2-layer, 4-head) ──")
    tf_model = TransformerDecoder(
        vocab_size, d_model=64, n_heads=4, n_layers=2, seq_len=SEQ_LEN)
    idx2 = list(range(len(train_seqs)))

    t0 = time.time()
    tf_bpc_curve = []
    for epoch in range(EPOCHS):
        random.shuffle(idx2)
        ep_bpc = 0.0
        for i in idx2[:N_TRAIN]:
            _, bpc = tf_model.train_step(train_seqs[i], lr=LR)
            ep_bpc += bpc
        avg = ep_bpc / N_TRAIN
        tf_bpc_curve.append(avg)
        print(f"  TF  epoch {epoch+1}/{EPOCHS}  train BPC={avg:.3f}")
    tf_train_t = time.time() - t0

    tf_test_bpc = 0.0
    for i in range(N_TEST):
        _, bpc = tf_model.train_step(test_seqs[i % len(test_seqs)], lr=0.0)
        tf_test_bpc += bpc
    tf_test_bpc /= N_TEST

    gen_ids = tf_model.generate(seed_ids, 200)
    tf_sample = seed_text + "".join(rvocab[i] for i in gen_ids)

    # ── Memory footprint (parameters × 4 bytes) ──
    lnn_mem = lnn_model.n_params * 4 / 1024
    tf_mem  = tf_model.n_params  * 4 / 1024

    # ── Summary ──
    print("\n" + "="*68)
    print(f"{'Model':<22} {'Test BPC':>9} {'Train(s)':>10} "
          f"{'Params':>9} {'Mem(KB)':>8}")
    print("-"*68)
    print(f"{'LNN (CTRNN, H=256)':<22} {lnn_test_bpc:>9.3f} "
          f"{lnn_train_t:>10.1f} {lnn_model.n_params:>9,} {lnn_mem:>8.1f}")
    print(f"{'Transformer (d=64)':<22} {tf_test_bpc:>9.3f} "
          f"{tf_train_t:>10.1f} {tf_model.n_params:>9,} {tf_mem:>8.1f}")
    print("="*68)

    print("\n── LNN sample ──")
    print(lnn_sample[:300])
    print("\n── Transformer sample ──")
    print(tf_sample[:300])

    # Save results
    import json
    os.makedirs("results", exist_ok=True)
    results = {
        "lnn":  {"test_bpc": lnn_test_bpc, "train_time": lnn_train_t,
                 "n_params": lnn_model.n_params, "mem_kb": lnn_mem,
                 "bpc_curve": lnn_bpc_curve, "sample": lnn_sample},
        "tf":   {"test_bpc": tf_test_bpc,  "train_time": tf_train_t,
                 "n_params": tf_model.n_params,  "mem_kb": tf_mem,
                 "bpc_curve": tf_bpc_curve,  "sample": tf_sample},
        "meta": {"epochs": EPOCHS, "n_train": N_TRAIN, "seq_len": SEQ_LEN,
                 "vocab_size": vocab_size, "lr": LR},
    }
    with open("results/shakespeare_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/shakespeare_results.json")

if __name__ == "__main__":
    main()
