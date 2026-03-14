#include "transformer.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * Layer Normalization
 * ---------------------------------------------------------------------------
 */
void layer_norm(float *out, const float *x,
                const float *gamma, const float *beta,
                int seq_len, int d_model, float eps)
{
    for (int i = 0; i < seq_len; i++) {
        const float *row = x + i * d_model;
        float *out_row   = out + i * d_model;

        /* mean */
        float mean = 0.0f;
        for (int j = 0; j < d_model; j++) mean += row[j];
        mean /= (float)d_model;

        /* variance */
        float var = 0.0f;
        for (int j = 0; j < d_model; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= (float)d_model;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* normalize, scale, shift */
        for (int j = 0; j < d_model; j++)
            out_row[j] = gamma[j] * (row[j] - mean) * inv_std + beta[j];
    }
}

/* ---------------------------------------------------------------------------
 * Matrix multiplication: out = A @ B  (row-major)
 * A [M x K], B [K x N], out [M x N]
 * ---------------------------------------------------------------------------
 */
void matmul(float *out, const float *A, const float *B, int M, int K, int N)
{
    memset(out, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++)
                out[i * N + j] += a_ik * B[k * N + j];
        }
    }
}

/* ---------------------------------------------------------------------------
 * Softmax in-place over each row of a [rows x cols] matrix
 * Numerically stable via max subtraction.
 * ---------------------------------------------------------------------------
 */
static void softmax_rows(float *x, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float *row = x + i * cols;

        float max_val = row[0];
        for (int j = 1; j < cols; j++)
            if (row[j] > max_val) max_val = row[j];

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        for (int j = 0; j < cols; j++) row[j] /= sum;
    }
}

/* ---------------------------------------------------------------------------
 * Multi-Head Attention
 * ---------------------------------------------------------------------------
 */
void multi_head_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          const float *W_q, const float *W_k,
                          const float *W_v, const float *W_o,
                          int seq_q, int seq_k,
                          int d_model, int num_heads, int causal_mask)
{
    int head_dim = d_model / num_heads;
    float scale  = 1.0f / sqrtf((float)head_dim);

    /* Project Q, K, V with learned weight matrices */
    float *Q_proj   = (float *)malloc((size_t)seq_q * d_model * sizeof(float));
    float *K_proj   = (float *)malloc((size_t)seq_k * d_model * sizeof(float));
    float *V_proj   = (float *)malloc((size_t)seq_k * d_model * sizeof(float));
    /* Accumulates per-head outputs before the output projection */
    float *attn_out = (float *)malloc((size_t)seq_q * d_model * sizeof(float));

    matmul(Q_proj, Q, W_q, seq_q, d_model, d_model);
    matmul(K_proj, K, W_k, seq_k, d_model, d_model);
    matmul(V_proj, V, W_v, seq_k, d_model, d_model);
    memset(attn_out, 0, (size_t)seq_q * d_model * sizeof(float));

    float *scores = (float *)malloc((size_t)seq_q * seq_k * sizeof(float));

    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;

        /* Scaled dot-product attention scores for this head */
        for (int i = 0; i < seq_q; i++) {
            for (int j = 0; j < seq_k; j++) {
                float dot = 0.0f;
                for (int k = 0; k < head_dim; k++)
                    dot += Q_proj[i * d_model + offset + k]
                         * K_proj[j * d_model + offset + k];
                scores[i * seq_k + j] = dot * scale;
            }
        }

        /* Causal mask: positions j > i cannot be attended to (set to -inf) */
        if (causal_mask) {
            for (int i = 0; i < seq_q; i++)
                for (int j = i + 1; j < seq_k; j++)
                    scores[i * seq_k + j] = -1e9f;
        }

        /* Softmax over key dimension */
        softmax_rows(scores, seq_q, seq_k);

        /* Weighted sum of values, written into the head's slice of attn_out */
        for (int i = 0; i < seq_q; i++) {
            for (int k = 0; k < head_dim; k++) {
                float val = 0.0f;
                for (int j = 0; j < seq_k; j++)
                    val += scores[i * seq_k + j] * V_proj[j * d_model + offset + k];
                attn_out[i * d_model + offset + k] = val;
            }
        }
    }

    free(scores);

    /* Output projection: out = concat(head_1,...,head_h) @ W_o */
    matmul(out, attn_out, W_o, seq_q, d_model, d_model);

    free(Q_proj);
    free(K_proj);
    free(V_proj);
    free(attn_out);
}

/* ---------------------------------------------------------------------------
 * Feed-Forward Network
 * FFN(x) = ReLU(x W1 + b1) W2 + b2
 * ---------------------------------------------------------------------------
 */
void ffn_forward(float *out, const float *x,
                 const float *W1, const float *b1,
                 const float *W2, const float *b2,
                 int seq_len, int d_model, int d_ff)
{
    float *hidden = (float *)malloc((size_t)seq_len * d_ff * sizeof(float));

    /* hidden = ReLU(x @ W1 + b1) */
    matmul(hidden, x, W1, seq_len, d_model, d_ff);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_ff; j++) {
            float v = hidden[i * d_ff + j] + b1[j];
            hidden[i * d_ff + j] = v > 0.0f ? v : 0.0f;   /* ReLU */
        }
    }

    /* out = hidden @ W2 + b2 */
    matmul(out, hidden, W2, seq_len, d_ff, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            out[i * d_model + j] += b2[j];

    free(hidden);
}

/* ---------------------------------------------------------------------------
 * Pre-LN Encoder Layer
 *   h   = x + MultiHeadAttn(LayerNorm(x), LayerNorm(x), LayerNorm(x))
 *   out = h + FFN(LayerNorm(h))
 * ---------------------------------------------------------------------------
 */
void encoder_layer(float *out, const float *x,
                   const float *norm1_gamma, const float *norm1_beta,
                   const float *W_q, const float *W_k,
                   const float *W_v, const float *W_o,
                   const float *norm2_gamma, const float *norm2_beta,
                   const float *W1, const float *b1,
                   const float *W2, const float *b2,
                   int seq_len, int d_model, int num_heads, int d_ff)
{
    size_t sz = (size_t)seq_len * d_model * sizeof(float);

    float *norm_x   = (float *)malloc(sz);
    float *attn_out = (float *)malloc(sz);
    float *h        = (float *)malloc(sz);
    float *ffn_out  = (float *)malloc(sz);

    /* --- Sublayer 1: self-attention with Pre-LN --- */
    layer_norm(norm_x, x, norm1_gamma, norm1_beta, seq_len, d_model, 1e-5f);
    multi_head_attention(attn_out,
                         norm_x, norm_x, norm_x,    /* Q=K=V (self-attn) */
                         W_q, W_k, W_v, W_o,
                         seq_len, seq_len, d_model, num_heads,
                         0 /* no causal mask */);
    for (int i = 0; i < seq_len * d_model; i++) h[i] = x[i] + attn_out[i];

    /* --- Sublayer 2: FFN with Pre-LN --- */
    layer_norm(norm_x, h, norm2_gamma, norm2_beta, seq_len, d_model, 1e-5f);
    ffn_forward(ffn_out, norm_x, W1, b1, W2, b2, seq_len, d_model, d_ff);
    for (int i = 0; i < seq_len * d_model; i++) out[i] = h[i] + ffn_out[i];

    free(norm_x);
    free(attn_out);
    free(h);
    free(ffn_out);
}

/* ---------------------------------------------------------------------------
 * Pre-LN Decoder Layer
 *   h1  = x  + MaskedMultiHeadAttn(LayerNorm(x),  LayerNorm(x),  LayerNorm(x))
 *   h2  = h1 + CrossAttn(LayerNorm(h1), enc_out, enc_out)
 *   out = h2 + FFN(LayerNorm(h2))
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
                   int d_model, int num_heads, int d_ff)
{
    size_t sz_tgt = (size_t)tgt_seq_len * d_model * sizeof(float);

    float *norm_x   = (float *)malloc(sz_tgt);
    float *attn_out = (float *)malloc(sz_tgt);
    float *h1       = (float *)malloc(sz_tgt);
    float *h2       = (float *)malloc(sz_tgt);
    float *ffn_out  = (float *)malloc(sz_tgt);

    /* --- Sublayer 1: masked self-attention with Pre-LN --- */
    layer_norm(norm_x, x, norm1_gamma, norm1_beta, tgt_seq_len, d_model, 1e-5f);
    multi_head_attention(attn_out,
                         norm_x, norm_x, norm_x,
                         self_W_q, self_W_k, self_W_v, self_W_o,
                         tgt_seq_len, tgt_seq_len, d_model, num_heads,
                         1 /* causal mask */);
    for (int i = 0; i < tgt_seq_len * d_model; i++) h1[i] = x[i] + attn_out[i];

    /* --- Sublayer 2: cross-attention with Pre-LN --- */
    /* Q comes from decoder (h1), K and V come from encoder (enc_out) */
    layer_norm(norm_x, h1, norm2_gamma, norm2_beta, tgt_seq_len, d_model, 1e-5f);
    multi_head_attention(attn_out,
                         norm_x, enc_out, enc_out,  /* Q=decoder, K=V=encoder */
                         cross_W_q, cross_W_k, cross_W_v, cross_W_o,
                         tgt_seq_len, src_seq_len, d_model, num_heads,
                         0 /* no causal mask on cross-attn */);
    for (int i = 0; i < tgt_seq_len * d_model; i++) h2[i] = h1[i] + attn_out[i];

    /* --- Sublayer 3: FFN with Pre-LN --- */
    layer_norm(norm_x, h2, norm3_gamma, norm3_beta, tgt_seq_len, d_model, 1e-5f);
    ffn_forward(ffn_out, norm_x, W1, b1, W2, b2, tgt_seq_len, d_model, d_ff);
    for (int i = 0; i < tgt_seq_len * d_model; i++) out[i] = h2[i] + ffn_out[i];

    free(norm_x);
    free(attn_out);
    free(h1);
    free(h2);
    free(ffn_out);
}

/* ---------------------------------------------------------------------------
 * Sinusoidal Positional Encoding
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 * ---------------------------------------------------------------------------
 */
void positional_encoding(float *out, int max_seq_len, int d_model)
{
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float div_term = powf(10000.0f, (float)i / (float)d_model);
            out[pos * d_model + i]     = sinf((float)pos / div_term);
            if (i + 1 < d_model)
                out[pos * d_model + i + 1] = cosf((float)pos / div_term);
        }
    }
}
