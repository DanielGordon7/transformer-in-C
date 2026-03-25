"""
UCI HAR Experiment: LNN vs Vanilla RNN
Compares on raw inertial sensor sequences (128 timesteps × 9 channels → 6 classes).
"""

import sys, os, time, math, random, struct, zipfile, urllib.request
import numpy as np
import lnn as lnn_module

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Download & load ────────────────────────────────────────────────────────

def generate_har_data(n_samples=2500, seed=SEED):
    """
    Synthetic HAR-like dataset: 128 timesteps × 9 sensor channels → 6 classes.
    Each class has a distinct temporal pattern mimicking real inertial signals:
      0 Walking      – periodic ~2 Hz oscillation, moderate amplitude
      1 Walk Up      – periodic with linear amplitude increase
      2 Walk Down    – periodic with linear amplitude decrease
      3 Sitting      – near-zero, low variance
      4 Standing     – near-zero, slightly higher variance
      5 Laying       – very low variance, different channel mix
    """
    rng = np.random.RandomState(seed)
    T, C = 128, 9
    X = np.zeros((n_samples, T, C), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int32)

    t = np.linspace(0, 1, T, dtype=np.float32)

    for i in range(n_samples):
        cls = i % N_CLASSES
        y[i] = cls
        noise = rng.randn(T, C).astype(np.float32) * 0.1

        if cls == 0:   # Walking – regular 2 Hz
            base = np.sin(2 * np.pi * 2.0 * t)[:, None] * np.array(
                [1.0, 0.6, 0.3, 0.8, 0.4, 0.2, 1.0, 0.6, 0.3])
        elif cls == 1:  # Walking upstairs – increasing amplitude
            amp = np.linspace(0.5, 1.5, T, dtype=np.float32)
            base = (amp * np.sin(2 * np.pi * 2.2 * t))[:, None] * np.array(
                [1.0, 0.7, 0.5, 0.9, 0.5, 0.3, 1.0, 0.7, 0.5])
        elif cls == 2:  # Walking downstairs – decreasing amplitude
            amp = np.linspace(1.5, 0.5, T, dtype=np.float32)
            base = (amp * np.sin(2 * np.pi * 1.8 * t))[:, None] * np.array(
                [1.0, 0.7, 0.5, 0.9, 0.5, 0.3, 1.0, 0.7, 0.5])
        elif cls == 3:  # Sitting – near-zero, mild drift
            base = (np.sin(2 * np.pi * 0.1 * t) * 0.05)[:, None] * np.ones(C)
            noise *= 0.3
        elif cls == 4:  # Standing – very slight sway
            base = (np.sin(2 * np.pi * 0.2 * t) * 0.1)[:, None] * np.ones(C)
            noise *= 0.4
        else:           # Laying – flat with very low noise
            base = (np.sin(2 * np.pi * 0.05 * t) * 0.02)[:, None] * np.array(
                [0.1, 1.0, 0.1, 0.1, 0.8, 0.1, 0.1, 1.0, 0.1])
            noise *= 0.15

        X[i] = base + noise

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]

CLASSES = ["Walking", "Walk Up", "Walk Down", "Sitting", "Standing", "Laying"]
N_CLASSES = 6
T_STEPS   = 128
N_CHAN    =  9

# ── Helpers ────────────────────────────────────────────────────────────────

def one_hot(label, n=N_CLASSES):
    v = np.zeros(n, dtype=np.float32)
    v[label] = 1.0
    return v

def accuracy(preds, labels):
    return float(np.mean(preds == labels))

def count_params_lnn(net):
    R, I, O = net.reservoir_size, net.input_size, net.output_size
    return R * I + R * R + O * R + R + O + R   # W_in+W_rec+W_out+b_rec+b_out+tau

# ── Vanilla RNN (numpy, BPTT) ──────────────────────────────────────────────

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=SEED):
        rng = np.random.RandomState(seed)
        s = 0.1
        self.W_in  = rng.randn(hidden_size, input_size).astype(np.float32)  * s
        self.W_h   = (rng.randn(hidden_size, hidden_size).astype(np.float32)
                      * 0.9 / math.sqrt(hidden_size))
        self.W_out = rng.randn(output_size, hidden_size).astype(np.float32) * s
        self.b_h   = np.zeros(hidden_size, dtype=np.float32)
        self.b_out = np.zeros(output_size, dtype=np.float32)
        self.h0    = np.zeros(hidden_size, dtype=np.float32)

    @property
    def n_params(self):
        return sum(w.size for w in
                   [self.W_in, self.W_h, self.W_out, self.b_h, self.b_out])

    def forward(self, X):
        """X: (T, I) → logits: (O,), stores h_seq for backward"""
        h = self.h0.copy()
        self.h_seq = [h]
        for t in range(len(X)):
            h = np.tanh(self.W_in @ X[t] + self.W_h @ h + self.b_h)
            self.h_seq.append(h)
        self._X = X
        return self.W_out @ h + self.b_out

    def train_step(self, X, y_oh, lr=1e-3):
        logits = self.forward(X)
        diff   = logits - y_oh
        loss   = float(np.dot(diff, diff) / N_CLASSES)
        dl_dy  = 2.0 * diff / N_CLASSES

        dW_out = np.outer(dl_dy, self.h_seq[-1])
        db_out = dl_dy.copy()
        dh     = self.W_out.T @ dl_dy

        dW_in = np.zeros_like(self.W_in)
        dW_h  = np.zeros_like(self.W_h)
        db_h  = np.zeros_like(self.b_h)

        for t in reversed(range(len(X))):
            dt_ = (1.0 - self.h_seq[t + 1] ** 2) * dh
            dW_in += np.outer(dt_, X[t])
            dW_h  += np.outer(dt_, self.h_seq[t])
            db_h  += dt_
            dh     = self.W_h.T @ dt_

        for g in [dW_in, dW_h, dW_out, db_h, db_out]:
            n = np.linalg.norm(g)
            if n > 5.0: g *= 5.0 / n

        self.W_in  -= lr * dW_in
        self.W_h   -= lr * dW_h
        self.W_out -= lr * dW_out
        self.b_h   -= lr * db_h
        self.b_out -= lr * db_out
        return loss

    def predict(self, X):
        return int(np.argmax(self.forward(X)))


# ── Training loops ─────────────────────────────────────────────────────────

def train_lnn(X_train, y_train, X_test, y_test,
              hidden=64, epochs=5, lr=1e-3, n_train=None, n_test=None):
    if n_train: X_train, y_train = X_train[:n_train], y_train[:n_train]
    if n_test:  X_test,  y_test  = X_test[:n_test],   y_test[:n_test]

    net = lnn_module.LiquidNN(N_CHAN, hidden, N_CLASSES, dt=0.1, ode_steps=1)
    idx = list(range(len(X_train)))

    t0 = time.time()
    for epoch in range(epochs):
        random.shuffle(idx)
        ep_loss = 0.0
        for i in idx:
            seq = X_train[i]            # (128, 9)
            flat = seq.flatten().tolist()
            tgt  = one_hot(y_train[i]).tolist()
            net.reset_state()
            ep_loss += net.train_sequence(flat, T_STEPS, tgt, lr=lr)
        print(f"  LNN  epoch {epoch+1}/{epochs}  loss={ep_loss/len(idx):.4f}")
    train_time = time.time() - t0

    # Evaluate
    preds = []
    t_inf = time.time()
    for i in range(len(X_test)):
        flat = X_test[i].flatten().tolist()
        net.reset_state()
        out = net.forward_sequence(flat, T_STEPS)
        preds.append(int(max(range(N_CLASSES), key=lambda k: out[k])))
    inf_time = (time.time() - t_inf) / len(X_test) * 1000  # ms per sample

    acc = accuracy(np.array(preds), y_test)
    params = count_params_lnn(net)
    return acc, train_time, inf_time, params


def train_rnn(X_train, y_train, X_test, y_test,
              hidden=64, epochs=5, lr=1e-3, n_train=None, n_test=None):
    if n_train: X_train, y_train = X_train[:n_train], y_train[:n_train]
    if n_test:  X_test,  y_test  = X_test[:n_test],   y_test[:n_test]

    net = VanillaRNN(N_CHAN, hidden, N_CLASSES)
    idx = list(range(len(X_train)))

    t0 = time.time()
    for epoch in range(epochs):
        random.shuffle(idx)
        ep_loss = 0.0
        for i in idx:
            net.h_seq = None
            ep_loss += net.train_step(X_train[i], one_hot(y_train[i]), lr=lr)
        print(f"  RNN  epoch {epoch+1}/{epochs}  loss={ep_loss/len(idx):.4f}")
    train_time = time.time() - t0

    preds = []
    t_inf = time.time()
    for i in range(len(X_test)):
        preds.append(net.predict(X_test[i]))
    inf_time = (time.time() - t_inf) / len(X_test) * 1000

    acc = accuracy(np.array(preds), y_test)
    return acc, train_time, inf_time, net.n_params


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic HAR-like dataset …")
    X_all, y_all = generate_har_data(n_samples=3000)
    split = 2400
    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]
    print(f"  train={len(X_train)}, test={len(X_test)}, "
          f"shape={X_train.shape[1:]}  classes={N_CLASSES}")

    # Normalise
    mu  = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1),  keepdims=True) + 1e-6
    X_train = (X_train - mu) / std
    X_test  = (X_test  - mu) / std

    HIDDEN = 64
    EPOCHS = 5
    LR     = 5e-4
    N_TRAIN = len(X_train)
    N_TEST  = len(X_test)

    print(f"\nRunning with {N_TRAIN} train / {N_TEST} test, "
          f"hidden={HIDDEN}, epochs={EPOCHS}\n")

    print("── LNN ──")
    lnn_acc, lnn_train_t, lnn_inf_t, lnn_params = train_lnn(
        X_train, y_train, X_test, y_test,
        hidden=HIDDEN, epochs=EPOCHS, lr=LR)

    print("\n── Vanilla RNN ──")
    rnn_acc, rnn_train_t, rnn_inf_t, rnn_params = train_rnn(
        X_train, y_train, X_test, y_test,
        hidden=HIDDEN, epochs=EPOCHS, lr=LR)

    # ── Print summary table ──────────────────────────────────────
    print("\n" + "="*62)
    print(f"{'Model':<18} {'Acc':>6} {'Train(s)':>10} {'Inf(ms)':>9} {'Params':>8}")
    print("-"*62)
    print(f"{'LNN (CTRNN)':<18} {lnn_acc*100:>5.1f}% {lnn_train_t:>10.1f} "
          f"{lnn_inf_t:>8.2f} {lnn_params:>8,}")
    print(f"{'Vanilla RNN':<18} {rnn_acc*100:>5.1f}% {rnn_train_t:>10.1f} "
          f"{rnn_inf_t:>8.2f} {rnn_params:>8,}")
    print("="*62)

    # Save for markdown
    results = {
        "lnn":  (lnn_acc,  lnn_train_t,  lnn_inf_t,  lnn_params),
        "rnn":  (rnn_acc,  rnn_train_t,  rnn_inf_t,  rnn_params),
        "meta": {"n_train": N_TRAIN, "n_test": N_TEST,
                 "hidden": HIDDEN, "epochs": EPOCHS, "lr": LR},
    }
    import json
    os.makedirs("results", exist_ok=True)
    with open("results/har_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/har_results.json")

if __name__ == "__main__":
    main()
