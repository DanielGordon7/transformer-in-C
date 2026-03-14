#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Transformer architecture (Pre-LN) in C
 *
 * Implements the encoder-decoder architecture from "Attention Is All You Need"
 * (Vaswani et al., 2017) with Pre-Layer Normalization (normalize before
 * attention/FFN sublayers) for better training stability.
 *
 * Pre-LN Encoder layer:
 *   h   = x + MultiHeadAttn(LayerNorm(x), LayerNorm(x), LayerNorm(x))
 *   out = h + FFN(LayerNorm(h))
 *
 * Pre-LN Decoder layer:
 *   h1  = x  + MaskedMultiHeadAttn(LayerNorm(x),  LayerNorm(x),  LayerNorm(x))
 *   h2  = h1 + CrossAttn(LayerNorm(h1), enc_out, enc_out)
 *   out = h2 + FFN(LayerNorm(h2))
 *
 * All matrices are stored in row-major order as flat float arrays.
 * ---------------------------------------------------------------------------
 */

/* ---------------------------------------------------------------------------
 * Layer Normalization
 *
 * Normalizes each token (row) independently:
 *   x_norm = (x - mean) / sqrt(var + eps)
 *   out    = gamma * x_norm + beta
 *
 * x, out : [seq_len x d_model]
 * gamma  : [d_model]  (scale)
 * beta   : [d_model]  (shift)
 * eps    : small constant for numerical stability (typically 1e-5)
 * ---------------------------------------------------------------------------
 */
void layer_norm(float *out, const float *x,
                const float *gamma, const float *beta,
                int seq_len, int d_model, float eps);

/* ---------------------------------------------------------------------------
 * Matrix multiplication: out = A @ B
 *
 * A   : [M x K]
 * B   : [K x N]
 * out : [M x N]  (caller-allocated, zeroed inside)
 * ---------------------------------------------------------------------------
 */
void matmul(float *out, const float *A, const float *B, int M, int K, int N);

/* ---------------------------------------------------------------------------
 * Multi-Head Attention
 *
 * Computes Attention(Q W_q, K W_k, V W_v) W_o with num_heads heads.
 * When causal_mask=1 each query position can only attend to positions <= itself
 * (used for decoder masked self-attention).
 *
 * Q    : [seq_q x d_model]
 * K, V : [seq_k x d_model]
 * W_q, W_k, W_v, W_o : [d_model x d_model]
 * out  : [seq_q x d_model]
 *
 * Requires d_model % num_heads == 0.
 * ---------------------------------------------------------------------------
 */
void multi_head_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          const float *W_q, const float *W_k,
                          const float *W_v, const float *W_o,
                          int seq_q, int seq_k,
                          int d_model, int num_heads, int causal_mask);

/* ---------------------------------------------------------------------------
 * Position-wise Feed-Forward Network
 *
 * FFN(x) = ReLU(x W1 + b1) W2 + b2
 *
 * x   : [seq_len x d_model]
 * W1  : [d_model x d_ff]
 * b1  : [d_ff]
 * W2  : [d_ff x d_model]
 * b2  : [d_model]
 * out : [seq_len x d_model]
 * ---------------------------------------------------------------------------
 */
void ffn_forward(float *out, const float *x,
                 const float *W1, const float *b1,
                 const float *W2, const float *b2,
                 int seq_len, int d_model, int d_ff);

/* ---------------------------------------------------------------------------
 * Single Pre-LN Encoder Layer
 *
 * x    : [seq_len x d_model]  (input)
 * out  : [seq_len x d_model]  (output, must not alias x)
 *
 * Sublayer weights:
 *   norm1_gamma/beta   : [d_model] — LN before self-attention
 *   W_q, W_k, W_v, W_o: [d_model x d_model] — self-attention projections
 *   norm2_gamma/beta   : [d_model] — LN before FFN
 *   W1 : [d_model x d_ff],  b1 : [d_ff]
 *   W2 : [d_ff x d_model],  b2 : [d_model]
 * ---------------------------------------------------------------------------
 */
void encoder_layer(float *out, const float *x,
                   const float *norm1_gamma, const float *norm1_beta,
                   const float *W_q, const float *W_k,
                   const float *W_v, const float *W_o,
                   const float *norm2_gamma, const float *norm2_beta,
                   const float *W1, const float *b1,
                   const float *W2, const float *b2,
                   int seq_len, int d_model, int num_heads, int d_ff);

/* ---------------------------------------------------------------------------
 * Single Pre-LN Decoder Layer
 *
 * x       : [tgt_seq_len x d_model]  (decoder input, must not alias out)
 * enc_out : [src_seq_len x d_model]  (encoder output, used for cross-attn)
 * out     : [tgt_seq_len x d_model]
 *
 * Sublayer weights:
 *   norm1_gamma/beta          : [d_model] — LN before masked self-attention
 *   self_W_q/k/v/o            : [d_model x d_model]
 *   norm2_gamma/beta          : [d_model] — LN before cross-attention
 *   cross_W_q/k/v/o           : [d_model x d_model]
 *   norm3_gamma/beta          : [d_model] — LN before FFN
 *   W1 : [d_model x d_ff],  b1 : [d_ff]
 *   W2 : [d_ff x d_model],  b2 : [d_model]
 * ---------------------------------------------------------------------------
 */
void decoder_layer(float *out, const float *x, const float *enc_out,
                   const float *norm1_gamma, const float *norm1_beta,
                   const float *self_W_q, const float *self_W_k,
                   const float *self_W_v, const float *self_W_o,
                   const float *norm2_gamma, const float *norm2_beta,
                   const float *cross_W_q, const float *cross_W_k,
                   const float *cross_W_v, const float *cross_W_o,
                   const float *norm3_gamma, const float *norm3_beta,
                   const float *W1, const float *b1,
                   const float *W2, const float *b2,
                   int tgt_seq_len, int src_seq_len,
                   int d_model, int num_heads, int d_ff);

/* ---------------------------------------------------------------------------
 * Sinusoidal Positional Encoding
 *
 * Fills out with standard sinusoidal position embeddings:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * out : [max_seq_len x d_model]
 * ---------------------------------------------------------------------------
 */
void positional_encoding(float *out, int max_seq_len, int d_model);

#endif /* TRANSFORMER_H */
