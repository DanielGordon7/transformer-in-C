#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

// Performs scaled dot-product self-attention
// Q, K, V are input matrices of shape [seq_len x d_model]
// Output is written to `out` of shape [seq_len x d_model]
void self_attention(
    const float *Q,
    const float *K,
    const float *V,
    float *out,
    int seq_len,
    int d_model
);

#endif
