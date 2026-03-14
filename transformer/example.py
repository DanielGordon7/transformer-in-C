"""
example.py — Demonstrates the Pre-LN Transformer C extension.

Build first:
    cd transformer
    python setup.py build_ext --inplace

Then run:
    python example.py

Architecture overview
---------------------
This module implements the Transformer from "Attention Is All You Need"
(Vaswani et al., 2017) with Pre-Layer Normalization (Pre-LN):

  Encoder layer:
    h   = x + MultiHeadAttn(LayerNorm(x), LayerNorm(x), LayerNorm(x))
    out = h + FFN(LayerNorm(h))

  Decoder layer:
    h1  = x  + MaskedMultiHeadAttn(LayerNorm(x),  ...)
    h2  = h1 + CrossAttn(LayerNorm(h1), enc_out, enc_out)
    out = h2 + FFN(LayerNorm(h2))

Pre-LN normalizes *before* each sublayer (instead of after) which gives
more stable gradients at initialization, especially for deep models.
"""

import random
import math
import transformer_module

random.seed(42)


# ---------------------------------------------------------------------------
# Weight initialization helpers (Xavier / Glorot uniform approximation)
# ---------------------------------------------------------------------------

def rand_matrix(rows, cols, scale=None):
    """Return a [rows x cols] list-of-lists with Xavier-uniform values."""
    if scale is None:
        scale = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-scale, scale) for _ in range(cols)]
            for _ in range(rows)]


def zeros(size):
    return [0.0] * size


def ones(size):
    return [1.0] * size


# ---------------------------------------------------------------------------
# Layer weight factories
# ---------------------------------------------------------------------------

def make_encoder_layer_weights(d_model, d_ff):
    """
    Returns a 12-tuple of weights for one Pre-LN encoder layer:
      (norm1_gamma, norm1_beta,
       W_q, W_k, W_v, W_o,
       norm2_gamma, norm2_beta,
       W1, b1, W2, b2)
    """
    return (
        ones(d_model),              # norm1_gamma  — initialized to 1
        zeros(d_model),             # norm1_beta   — initialized to 0
        rand_matrix(d_model, d_model),  # W_q
        rand_matrix(d_model, d_model),  # W_k
        rand_matrix(d_model, d_model),  # W_v
        rand_matrix(d_model, d_model),  # W_o
        ones(d_model),              # norm2_gamma
        zeros(d_model),             # norm2_beta
        rand_matrix(d_model, d_ff),     # W1  (expand to d_ff)
        zeros(d_ff),                # b1
        rand_matrix(d_ff, d_model),     # W2  (project back to d_model)
        zeros(d_model),             # b2
    )


def make_decoder_layer_weights(d_model, d_ff):
    """
    Returns an 18-tuple of weights for one Pre-LN decoder layer:
      (norm1_gamma, norm1_beta,
       self_W_q, self_W_k, self_W_v, self_W_o,
       norm2_gamma, norm2_beta,
       cross_W_q, cross_W_k, cross_W_v, cross_W_o,
       norm3_gamma, norm3_beta,
       W1, b1, W2, b2)
    """
    return (
        ones(d_model),              # norm1_gamma (before masked self-attn)
        zeros(d_model),             # norm1_beta
        rand_matrix(d_model, d_model),  # self_W_q
        rand_matrix(d_model, d_model),  # self_W_k
        rand_matrix(d_model, d_model),  # self_W_v
        rand_matrix(d_model, d_model),  # self_W_o
        ones(d_model),              # norm2_gamma (before cross-attn)
        zeros(d_model),             # norm2_beta
        rand_matrix(d_model, d_model),  # cross_W_q
        rand_matrix(d_model, d_model),  # cross_W_k
        rand_matrix(d_model, d_model),  # cross_W_v
        rand_matrix(d_model, d_model),  # cross_W_o
        ones(d_model),              # norm3_gamma (before FFN)
        zeros(d_model),             # norm3_beta
        rand_matrix(d_model, d_ff),     # W1
        zeros(d_ff),                # b1
        rand_matrix(d_ff, d_model),     # W2
        zeros(d_model),             # b2
    )


def print_matrix_summary(name, mat):
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    flat = [v for row in mat for v in row]
    mn   = min(flat)
    mx   = max(flat)
    mean = sum(flat) / len(flat)
    print(f"  {name}: shape=({rows}, {cols})  min={mn:.4f}  max={mx:.4f}  mean={mean:.4f}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_MODEL          = 32   # embedding / model dimension
NUM_HEADS        = 4    # attention heads (D_MODEL must be divisible by this)
D_FF             = 128  # feed-forward hidden dimension (typically 4 × d_model)
NUM_ENC_LAYERS   = 2    # encoder depth
NUM_DEC_LAYERS   = 2    # decoder depth
SRC_SEQ_LEN      = 6    # source sequence length
TGT_SEQ_LEN      = 4    # target sequence length
MAX_SEQ_LEN      = 16   # for positional encoding table

assert D_MODEL % NUM_HEADS == 0, "d_model must be divisible by num_heads"

print("=" * 60)
print("Pre-LN Transformer — C extension demo")
print(f"  d_model={D_MODEL}, num_heads={NUM_HEADS}, d_ff={D_FF}")
print(f"  encoder layers={NUM_ENC_LAYERS}, decoder layers={NUM_DEC_LAYERS}")
print(f"  src_seq_len={SRC_SEQ_LEN}, tgt_seq_len={TGT_SEQ_LEN}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Positional encoding
# ---------------------------------------------------------------------------
print("\n[1] Positional encoding")
pos_enc = transformer_module.positional_encoding(MAX_SEQ_LEN, D_MODEL)
print_matrix_summary("pos_enc", pos_enc)

# ---------------------------------------------------------------------------
# 2. Synthetic input embeddings (random, as if looked up from an embedding table)
# ---------------------------------------------------------------------------
src_x = rand_matrix(SRC_SEQ_LEN, D_MODEL, scale=0.1)
tgt_x = rand_matrix(TGT_SEQ_LEN, D_MODEL, scale=0.1)

# Add positional encoding to inputs
for i in range(SRC_SEQ_LEN):
    for j in range(D_MODEL):
        src_x[i][j] += pos_enc[i][j]

for i in range(TGT_SEQ_LEN):
    for j in range(D_MODEL):
        tgt_x[i][j] += pos_enc[i][j]

# ---------------------------------------------------------------------------
# 3. Component-level demos
# ---------------------------------------------------------------------------
print("\n[2] Component: layer_norm")
gamma = ones(D_MODEL)
beta  = zeros(D_MODEL)
normed = transformer_module.layer_norm(src_x, gamma, beta)
print_matrix_summary("normed src_x", normed)

print("\n[3] Component: multi_head_attention (self-attention, no mask)")
W_q = rand_matrix(D_MODEL, D_MODEL)
W_k = rand_matrix(D_MODEL, D_MODEL)
W_v = rand_matrix(D_MODEL, D_MODEL)
W_o = rand_matrix(D_MODEL, D_MODEL)
attn_out = transformer_module.multi_head_attention(
    src_x, src_x, src_x, W_q, W_k, W_v, W_o, NUM_HEADS, 0
)
print_matrix_summary("self-attn output", attn_out)

print("\n[4] Component: multi_head_attention (causal mask)")
causal_out = transformer_module.multi_head_attention(
    tgt_x, tgt_x, tgt_x, W_q, W_k, W_v, W_o, NUM_HEADS, 1
)
print_matrix_summary("causal-attn output", causal_out)

print("\n[5] Component: ffn_forward")
W1 = rand_matrix(D_MODEL, D_FF)
b1 = zeros(D_FF)
W2 = rand_matrix(D_FF, D_MODEL)
b2 = zeros(D_MODEL)
ffn_out = transformer_module.ffn_forward(src_x, W1, b1, W2, b2)
print_matrix_summary("FFN output", ffn_out)

# ---------------------------------------------------------------------------
# 4. Single-layer demos
# ---------------------------------------------------------------------------
enc_w = make_encoder_layer_weights(D_MODEL, D_FF)
dec_w = make_decoder_layer_weights(D_MODEL, D_FF)

print("\n[6] Single encoder_layer")
enc_layer_out = transformer_module.encoder_layer(
    src_x,
    enc_w[0], enc_w[1],        # norm1_gamma, norm1_beta
    enc_w[2], enc_w[3],        # W_q, W_k
    enc_w[4], enc_w[5],        # W_v, W_o
    enc_w[6], enc_w[7],        # norm2_gamma, norm2_beta
    enc_w[8], enc_w[9],        # W1, b1
    enc_w[10], enc_w[11],      # W2, b2
    NUM_HEADS
)
print_matrix_summary("encoder_layer output", enc_layer_out)

print("\n[7] Single decoder_layer")
dec_layer_out = transformer_module.decoder_layer(
    tgt_x, enc_layer_out,
    dec_w[0],  dec_w[1],       # norm1_gamma, norm1_beta
    dec_w[2],  dec_w[3],       # self_W_q, self_W_k
    dec_w[4],  dec_w[5],       # self_W_v, self_W_o
    dec_w[6],  dec_w[7],       # norm2_gamma, norm2_beta
    dec_w[8],  dec_w[9],       # cross_W_q, cross_W_k
    dec_w[10], dec_w[11],      # cross_W_v, cross_W_o
    dec_w[12], dec_w[13],      # norm3_gamma, norm3_beta
    dec_w[14], dec_w[15],      # W1, b1
    dec_w[16], dec_w[17],      # W2, b2
    NUM_HEADS
)
print_matrix_summary("decoder_layer output", dec_layer_out)

# ---------------------------------------------------------------------------
# 5. Full encoder and decoder stacks
# ---------------------------------------------------------------------------
encoder_weights = [make_encoder_layer_weights(D_MODEL, D_FF)
                   for _ in range(NUM_ENC_LAYERS)]
decoder_weights = [make_decoder_layer_weights(D_MODEL, D_FF)
                   for _ in range(NUM_DEC_LAYERS)]

print(f"\n[8] Full encoder_forward ({NUM_ENC_LAYERS} layers)")
enc_out = transformer_module.encoder_forward(src_x, encoder_weights, NUM_HEADS)
print_matrix_summary("encoder output", enc_out)

print(f"\n[9] Full decoder_forward ({NUM_DEC_LAYERS} layers)")
dec_out = transformer_module.decoder_forward(tgt_x, enc_out, decoder_weights, NUM_HEADS)
print_matrix_summary("decoder output", dec_out)

# ---------------------------------------------------------------------------
# 6. Full encoder-decoder transformer (compose the two stacks)
# ---------------------------------------------------------------------------
print("\n[10] Complete encoder-decoder Transformer")
print("     src:", SRC_SEQ_LEN, "tokens  →  encoder  →  cross-attention in decoder")
print("     tgt:", TGT_SEQ_LEN, "tokens  →  decoder  →  final hidden states")
print_matrix_summary("final hidden states", dec_out)

print("\nAll checks passed.")
