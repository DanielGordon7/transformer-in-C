"""
Example usage of the self_attention_module.

Build first:
    cd attention
    python setup.py build_ext --inplace

Then run:
    python example.py
"""

import self_attention_module
import random

def make_matrix(seq_len, d_model):
    return [[random.random() for _ in range(d_model)] for _ in range(seq_len)]

if __name__ == "__main__":
    seq_len = 4
    d_model = 8

    Q = make_matrix(seq_len, d_model)
    K = make_matrix(seq_len, d_model)
    V = make_matrix(seq_len, d_model)

    print(f"Running self-attention with seq_len={seq_len}, d_model={d_model}\n")
    output = self_attention_module.self_attention(Q, K, V)

    print("Output matrix:")
    for i, row in enumerate(output):
        formatted = [f"{x:.4f}" for x in row]
        print(f"  Row {i}: {formatted}")
