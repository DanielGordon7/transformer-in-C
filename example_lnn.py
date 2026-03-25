"""
Liquid Neural Network – two demo tasks
  1. Sine-wave prediction  (regression on time-series)
  2. MNIST digit classification (sequential: one 28-pixel row per step)
"""

import math, random, sys, os
import lnn

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1 – Sine-wave next-step prediction
# ═════════════════════════════════════════════════════════════════════════════
def task_sine():
    print("=" * 60)
    print("TASK 1: Sine-wave next-step prediction")
    print("=" * 60)

    SEQ_LEN  = 200
    N_EPOCHS = 30
    LR       = 3e-3

    # input: (sin(t),) → target: (sin(t+1),)
    def make_seq(n=SEQ_LEN, freq=0.1):
        xs = [math.sin(2 * math.pi * freq * t) for t in range(n + 1)]
        return xs

    model = lnn.LiquidNN(
        input_size=1, reservoir_size=64, output_size=1,
        dt=0.1, ode_steps=4
    )

    seq = make_seq()
    for epoch in range(N_EPOCHS):
        model.reset_state()
        total_loss = 0.0
        for t in range(SEQ_LEN):
            inp    = [seq[t]]
            target = [seq[t + 1]]
            loss   = model.train_step(inp, target, lr=LR)
            total_loss += loss
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{N_EPOCHS}  avg_loss={total_loss/SEQ_LEN:.6f}")

    # evaluation
    model.reset_state()
    errors = []
    for t in range(SEQ_LEN):
        pred = model.forward([seq[t]])[0]
        errors.append(abs(pred - seq[t + 1]))
    print(f"\n  MAE on training sequence: {sum(errors)/len(errors):.6f}")
    print(f"  First 5 predictions vs targets:")
    model.reset_state()
    for t in range(5):
        pred = model.forward([seq[t]])[0]
        print(f"    t={t}  pred={pred:+.4f}  true={seq[t+1]:+.4f}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2 – MNIST digit classification (sequential)
# ═════════════════════════════════════════════════════════════════════════════

def download_mnist():
    """Download MNIST via sklearn (fetch_openml) or fallback mirrors."""
    # Try sklearn first – no network gymnastics needed
    try:
        from sklearn.datasets import fetch_openml
        print("  Fetching MNIST via sklearn/OpenML (may take a moment) …")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype("float32") / 255.0   # (70000, 784)
        y = mnist.target.astype(int)
        return X, y, 28, 28
    except Exception as e:
        print(f"  sklearn fetch failed: {e}")

    # Fallback: try alternative MNIST mirror
    import urllib.request, gzip, struct
    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    files = {
        "train-images-idx3-ubyte.gz": "train-images",
        "train-labels-idx1-ubyte.gz": "train-labels",
        "t10k-images-idx3-ubyte.gz":  "test-images",
        "t10k-labels-idx1-ubyte.gz":  "test-labels",
    }
    os.makedirs("mnist_data", exist_ok=True)
    for fname, label in files.items():
        out = os.path.join("mnist_data", label)
        if os.path.exists(out):
            continue
        downloaded = False
        for base in mirrors:
            try:
                print(f"  Downloading {fname} from {base} …")
                urllib.request.urlretrieve(base + fname, out + ".gz")
                with gzip.open(out + ".gz", "rb") as f_in, open(out, "wb") as f_out:
                    f_out.write(f_in.read())
                os.remove(out + ".gz")
                downloaded = True
                break
            except Exception as e:
                print(f"    failed: {e}")
        if not downloaded:
            return None
    return "mnist_data"


def load_mnist(path):
    import struct
    def read_images(fn):
        with open(fn, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = f.read()
        imgs = []
        sz = rows * cols
        for i in range(n):
            chunk = data[i * sz:(i + 1) * sz]
            imgs.append([b / 255.0 for b in chunk])
        return imgs, rows, cols

    def read_labels(fn):
        with open(fn, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            return list(f.read())

    train_x, rows, cols = read_images(os.path.join(path, "train-images"))
    train_y = read_labels(os.path.join(path, "train-labels"))
    test_x, _, _ = read_images(os.path.join(path, "test-images"))
    test_y = read_labels(os.path.join(path, "test-labels"))
    return train_x, train_y, test_x, test_y, rows, cols


def one_hot(label, n=10):
    v = [0.0] * n
    v[label] = 1.0
    return v


def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])


def task_mnist():
    print("=" * 60)
    print("TASK 2: MNIST classification (sequential, row-by-row)")
    print("=" * 60)

    result = download_mnist()
    if result is None:
        print("  Skipping MNIST (download failed).")
        return

    # sklearn path returns (X, y, rows, cols); legacy path returns directory string
    if isinstance(result, str):
        print("  Loading MNIST …")
        train_x, train_y, test_x, test_y, rows, cols = load_mnist(result)
    else:
        X, y, rows, cols = result
        n_train = 60000
        train_x = [list(X[i]) for i in range(n_train)]
        train_y = list(y[:n_train])
        test_x  = [list(X[i]) for i in range(n_train, len(X))]
        test_y  = list(y[n_train:])

    print(f"  train={len(train_x)}, test={len(test_x)}, image={rows}x{cols}")

    # Sequential: feed one row (28 pixels) per timestep; classify after last row
    INPUT   = cols      # 28
    HIDDEN  = 128
    OUTPUT  = 10
    LR      = 1e-3
    EPOCHS  = 3
    # Use a subset for speed; remove the slice to use full 60 k
    N_TRAIN = 5000
    N_TEST  = 1000

    model = lnn.LiquidNN(INPUT, HIDDEN, OUTPUT, dt=0.1, ode_steps=3)

    indices = list(range(len(train_x)))
    random.shuffle(indices)

    for epoch in range(EPOCHS):
        random.shuffle(indices)
        total_loss = 0.0
        correct    = 0

        for idx in indices[:N_TRAIN]:
            img = train_x[idx]
            lbl = train_y[idx]
            target = one_hot(lbl)

            model.reset_state()
            # feed rows one by one; only train on the last row
            for row_i in range(rows - 1):
                row = img[row_i * cols:(row_i + 1) * cols]
                model.forward(row)       # update state, no gradient
            # last row: train step
            last_row = img[(rows - 1) * cols:]
            loss = model.train_step(last_row, target, lr=LR)
            total_loss += loss

            # quick accuracy check with current output
            model.reset_state()
            out = None
            for row_i in range(rows):
                row = img[row_i * cols:(row_i + 1) * cols]
                out = model.forward(row)
            if argmax(out) == lbl:
                correct += 1

        acc = correct / N_TRAIN * 100
        avg_loss = total_loss / N_TRAIN
        print(f"  epoch {epoch+1}/{EPOCHS}  loss={avg_loss:.4f}  train_acc={acc:.1f}%")

    # test
    correct = 0
    for idx in range(N_TEST):
        img = test_x[idx]
        lbl = test_y[idx]
        model.reset_state()
        out = None
        for row_i in range(rows):
            row = img[row_i * cols:(row_i + 1) * cols]
            out = model.forward(row)
        if argmax(out) == lbl:
            correct += 1

    print(f"\n  Test accuracy ({N_TEST} samples): {correct/N_TEST*100:.1f}%")
    model.save("lnn_mnist.bin")
    print("  Model saved to lnn_mnist.bin")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    task_sine()
    task_mnist()
