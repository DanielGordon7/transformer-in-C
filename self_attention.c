#include "self_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float dot(const float *a, const float *b, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) sum += a[i] * b[i];
    return sum;
}

static void softmax(float *x, int len) {
    // cal culate max to substract from all values to avoid overflow in exponentiation
    float max_val = x[0];
    for (int i = 1; i < len; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < len; i++) x[i] /= sum;
}

// Scaled dot-product self-attention
// Q, K, V: row-major matrices of shape [seq_len x d_model]
// out:     output matrix of shape [seq_len x d_model]
void self_attention(const float *Q, const float *K, const float *V,
                    float *out, int seq_len, int d_model) {
    float scale = 1.0f / sqrtf((float)d_model);
    float *scores = (float *)malloc(seq_len * seq_len * sizeof(float));

    // initialize out to 0 (memory was allocated by caller)
    memset(out, 0, seq_len * d_model * sizeof(float));

    // Compute attention scores: scores[i][j] = dot(Q[i], K[j]) * scale
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            scores[i * seq_len + j] = dot(
                Q + i * d_model,
                K + j * d_model,
                d_model
            ) * scale;
        }
        // Softmax over each row
        softmax(scores + i * seq_len, seq_len);
    }

    // Compute output: out = scores @ V
    // scores: (seq_len, seq_len) shape
    // V: (seq_len, d_model) shape
    // out: (seq_len, d_model) shape
    // first loop: num rows of scores
    // middle loop: shared dim of seq_len and V
    // last loop: num cols of V

    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            for (int k = 0; k < seq_len; k++) {
                out[i * d_model + j] += scores[i * seq_len + k] * V[k * d_model + j];
            }
        }
    }

    free(scores);
}
