/*
 * transformer_module.c — Python/C API bindings for the Pre-LN Transformer.
 *
 * Exposed Python functions
 * ------------------------
 * Component-level (useful for testing and custom architectures):
 *   layer_norm(x, gamma, beta)                       -> [[float]]
 *   multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads, causal_mask=0)
 *                                                     -> [[float]]
 *   ffn_forward(x, W1, b1, W2, b2)                   -> [[float]]
 *   encoder_layer(x, norm1_gamma, norm1_beta,
 *                 W_q, W_k, W_v, W_o,
 *                 norm2_gamma, norm2_beta,
 *                 W1, b1, W2, b2, num_heads)          -> [[float]]
 *   decoder_layer(x, enc_out,
 *                 norm1_gamma, norm1_beta,
 *                 self_W_q, self_W_k, self_W_v, self_W_o,
 *                 norm2_gamma, norm2_beta,
 *                 cross_W_q, cross_W_k, cross_W_v, cross_W_o,
 *                 norm3_gamma, norm3_beta,
 *                 W1, b1, W2, b2, num_heads)          -> [[float]]
 *   positional_encoding(max_seq_len, d_model)         -> [[float]]
 *
 * Stack-level (run multiple layers in one call):
 *   encoder_forward(x, layer_weights, num_heads)      -> [[float]]
 *   decoder_forward(x, enc_out, layer_weights, num_heads) -> [[float]]
 *
 * Data formats
 * ------------
 * All matrices are passed as Python list-of-lists (nested lists).
 * All 1-D vectors (gamma, beta, biases) are plain Python lists of floats.
 *
 * layer_weights for encoder_forward is a list of 12-tuples, one per layer:
 *   (norm1_gamma, norm1_beta,
 *    W_q, W_k, W_v, W_o,
 *    norm2_gamma, norm2_beta,
 *    W1, b1, W2, b2)
 *
 * layer_weights for decoder_forward is a list of 18-tuples, one per layer:
 *   (norm1_gamma, norm1_beta,
 *    self_W_q, self_W_k, self_W_v, self_W_o,
 *    norm2_gamma, norm2_beta,
 *    cross_W_q, cross_W_k, cross_W_v, cross_W_o,
 *    norm3_gamma, norm3_beta,
 *    W1, b1, W2, b2)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "transformer.h"
#include <stdlib.h>
#include <string.h>

/* ===========================================================================
 * Helper utilities
 * =========================================================================*/

/* Extract a 1-D Python list of floats into a newly-allocated C array.
 * Returns NULL and sets a Python exception on error. */
static float *extract_1d(PyObject *lst, int size)
{
    if (!PyList_Check(lst) || (int)PyList_Size(lst) != size) {
        PyErr_Format(PyExc_ValueError,
                     "expected a list of length %d", size);
        return NULL;
    }
    float *arr = (float *)malloc((size_t)size * sizeof(float));
    if (!arr) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < size; i++) {
        PyObject *item = PyList_GET_ITEM(lst, i);
        double v = PyFloat_AsDouble(item);
        if (v == -1.0 && PyErr_Occurred()) { free(arr); return NULL; }
        arr[i] = (float)v;
    }
    return arr;
}

/* Extract a 2-D Python list-of-lists into a newly-allocated row-major array.
 * Returns NULL and sets a Python exception on error. */
static float *extract_2d(PyObject *lst, int rows, int cols)
{
    if (!PyList_Check(lst) || (int)PyList_Size(lst) != rows) {
        PyErr_Format(PyExc_ValueError,
                     "expected a list of %d rows", rows);
        return NULL;
    }
    float *arr = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!arr) { PyErr_NoMemory(); return NULL; }
    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_GET_ITEM(lst, i);
        if (!PyList_Check(row) || (int)PyList_Size(row) != cols) {
            PyErr_Format(PyExc_ValueError,
                         "row %d must be a list of %d floats", i, cols);
            free(arr); return NULL;
        }
        for (int j = 0; j < cols; j++) {
            PyObject *item = PyList_GET_ITEM(row, j);
            double v = PyFloat_AsDouble(item);
            if (v == -1.0 && PyErr_Occurred()) { free(arr); return NULL; }
            arr[i * cols + j] = (float)v;
        }
    }
    return arr;
}

/* Pack a flat C array into a Python list-of-lists [rows x cols]. */
static PyObject *pack_2d(const float *arr, int rows, int cols)
{
    PyObject *result = PyList_New(rows);
    if (!result) return NULL;
    for (int i = 0; i < rows; i++) {
        PyObject *row = PyList_New(cols);
        if (!row) { Py_DECREF(result); return NULL; }
        for (int j = 0; j < cols; j++) {
            PyObject *val = PyFloat_FromDouble((double)arr[i * cols + j]);
            if (!val) { Py_DECREF(row); Py_DECREF(result); return NULL; }
            PyList_SET_ITEM(row, j, val);
        }
        PyList_SET_ITEM(result, i, row);
    }
    return result;
}

/* ===========================================================================
 * layer_norm(x, gamma, beta) -> [[float]]
 * =========================================================================*/
static PyObject *py_layer_norm(PyObject *self, PyObject *args)
{
    PyObject *py_x, *py_gamma, *py_beta;
    if (!PyArg_ParseTuple(args, "OOO", &py_x, &py_gamma, &py_beta))
        return NULL;

    if (!PyList_Check(py_x) || PyList_Size(py_x) == 0) {
        PyErr_SetString(PyExc_ValueError, "x must be a non-empty list");
        return NULL;
    }
    int seq_len = (int)PyList_Size(py_x);
    int d_model = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));

    float *x     = extract_2d(py_x,    seq_len, d_model);
    float *gamma = extract_1d(py_gamma, d_model);
    float *beta  = extract_1d(py_beta,  d_model);
    float *out   = NULL;
    PyObject *result = NULL;

    if (!x || !gamma || !beta) goto cleanup;

    out = (float *)malloc((size_t)seq_len * d_model * sizeof(float));
    if (!out) { PyErr_NoMemory(); goto cleanup; }

    layer_norm(out, x, gamma, beta, seq_len, d_model, 1e-5f);
    result = pack_2d(out, seq_len, d_model);

cleanup:
    free(x); free(gamma); free(beta); free(out);
    return result;
}

/* ===========================================================================
 * multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads,
 *                      causal_mask=0) -> [[float]]
 * =========================================================================*/
static PyObject *py_multi_head_attention(PyObject *self, PyObject *args)
{
    PyObject *py_Q, *py_K, *py_V;
    PyObject *py_Wq, *py_Wk, *py_Wv, *py_Wo;
    int num_heads, causal_mask = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOi|i",
                          &py_Q, &py_K, &py_V,
                          &py_Wq, &py_Wk, &py_Wv, &py_Wo,
                          &num_heads, &causal_mask))
        return NULL;

    int seq_q   = (int)PyList_Size(py_Q);
    int seq_k   = (int)PyList_Size(py_K);
    int d_model = (int)PyList_Size(PyList_GET_ITEM(py_Q, 0));

    if (d_model % num_heads != 0) {
        PyErr_SetString(PyExc_ValueError, "d_model must be divisible by num_heads");
        return NULL;
    }

    float *Q   = extract_2d(py_Q,  seq_q,   d_model);
    float *K   = extract_2d(py_K,  seq_k,   d_model);
    float *V   = extract_2d(py_V,  seq_k,   d_model);
    float *W_q = extract_2d(py_Wq, d_model, d_model);
    float *W_k = extract_2d(py_Wk, d_model, d_model);
    float *W_v = extract_2d(py_Wv, d_model, d_model);
    float *W_o = extract_2d(py_Wo, d_model, d_model);
    float *out = NULL;
    PyObject *result = NULL;

    if (!Q || !K || !V || !W_q || !W_k || !W_v || !W_o) goto cleanup;

    out = (float *)malloc((size_t)seq_q * d_model * sizeof(float));
    if (!out) { PyErr_NoMemory(); goto cleanup; }

    multi_head_attention(out, Q, K, V, W_q, W_k, W_v, W_o,
                         seq_q, seq_k, d_model, num_heads, causal_mask);
    result = pack_2d(out, seq_q, d_model);

cleanup:
    free(Q); free(K); free(V);
    free(W_q); free(W_k); free(W_v); free(W_o);
    free(out);
    return result;
}

/* ===========================================================================
 * ffn_forward(x, W1, b1, W2, b2) -> [[float]]
 * =========================================================================*/
static PyObject *py_ffn_forward(PyObject *self, PyObject *args)
{
    PyObject *py_x, *py_W1, *py_b1, *py_W2, *py_b2;
    if (!PyArg_ParseTuple(args, "OOOOO", &py_x, &py_W1, &py_b1, &py_W2, &py_b2))
        return NULL;

    int seq_len = (int)PyList_Size(py_x);
    int d_model = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));
    int d_ff    = (int)PyList_Size(py_b1);

    float *x   = extract_2d(py_x,  seq_len, d_model);
    float *W1  = extract_2d(py_W1, d_model, d_ff);
    float *b1  = extract_1d(py_b1, d_ff);
    float *W2  = extract_2d(py_W2, d_ff,    d_model);
    float *b2  = extract_1d(py_b2, d_model);
    float *out = NULL;
    PyObject *result = NULL;

    if (!x || !W1 || !b1 || !W2 || !b2) goto cleanup;

    out = (float *)malloc((size_t)seq_len * d_model * sizeof(float));
    if (!out) { PyErr_NoMemory(); goto cleanup; }

    ffn_forward(out, x, W1, b1, W2, b2, seq_len, d_model, d_ff);
    result = pack_2d(out, seq_len, d_model);

cleanup:
    free(x); free(W1); free(b1); free(W2); free(b2); free(out);
    return result;
}

/* ===========================================================================
 * encoder_layer(x,
 *               norm1_gamma, norm1_beta,
 *               W_q, W_k, W_v, W_o,
 *               norm2_gamma, norm2_beta,
 *               W1, b1, W2, b2,
 *               num_heads) -> [[float]]
 * =========================================================================*/
static PyObject *py_encoder_layer(PyObject *self, PyObject *args)
{
    PyObject *py_x;
    PyObject *py_n1g, *py_n1b;
    PyObject *py_Wq, *py_Wk, *py_Wv, *py_Wo;
    PyObject *py_n2g, *py_n2b;
    PyObject *py_W1, *py_b1, *py_W2, *py_b2;
    int num_heads;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOi",
                          &py_x,
                          &py_n1g, &py_n1b,
                          &py_Wq, &py_Wk, &py_Wv, &py_Wo,
                          &py_n2g, &py_n2b,
                          &py_W1, &py_b1, &py_W2, &py_b2,
                          &num_heads))
        return NULL;

    int seq_len = (int)PyList_Size(py_x);
    int d_model = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));
    int d_ff    = (int)PyList_Size(py_b1);

    float *x       = extract_2d(py_x,  seq_len, d_model);
    float *n1g     = extract_1d(py_n1g, d_model);
    float *n1b     = extract_1d(py_n1b, d_model);
    float *W_q     = extract_2d(py_Wq,  d_model, d_model);
    float *W_k     = extract_2d(py_Wk,  d_model, d_model);
    float *W_v     = extract_2d(py_Wv,  d_model, d_model);
    float *W_o     = extract_2d(py_Wo,  d_model, d_model);
    float *n2g     = extract_1d(py_n2g, d_model);
    float *n2b     = extract_1d(py_n2b, d_model);
    float *W1      = extract_2d(py_W1,  d_model, d_ff);
    float *b1      = extract_1d(py_b1,  d_ff);
    float *W2      = extract_2d(py_W2,  d_ff,    d_model);
    float *b2      = extract_1d(py_b2,  d_model);
    float *out     = NULL;
    PyObject *result = NULL;

    if (!x || !n1g || !n1b || !W_q || !W_k || !W_v || !W_o ||
        !n2g || !n2b || !W1 || !b1 || !W2 || !b2) goto cleanup;

    out = (float *)malloc((size_t)seq_len * d_model * sizeof(float));
    if (!out) { PyErr_NoMemory(); goto cleanup; }

    encoder_layer(out, x,
                  n1g, n1b, W_q, W_k, W_v, W_o,
                  n2g, n2b, W1, b1, W2, b2,
                  seq_len, d_model, num_heads, d_ff);
    result = pack_2d(out, seq_len, d_model);

cleanup:
    free(x); free(n1g); free(n1b);
    free(W_q); free(W_k); free(W_v); free(W_o);
    free(n2g); free(n2b);
    free(W1); free(b1); free(W2); free(b2);
    free(out);
    return result;
}

/* ===========================================================================
 * decoder_layer(x, enc_out,
 *               norm1_gamma, norm1_beta,
 *               self_W_q, self_W_k, self_W_v, self_W_o,
 *               norm2_gamma, norm2_beta,
 *               cross_W_q, cross_W_k, cross_W_v, cross_W_o,
 *               norm3_gamma, norm3_beta,
 *               W1, b1, W2, b2,
 *               num_heads) -> [[float]]
 * =========================================================================*/
static PyObject *py_decoder_layer(PyObject *self, PyObject *args)
{
    PyObject *py_x, *py_enc;
    PyObject *py_n1g, *py_n1b;
    PyObject *py_sWq, *py_sWk, *py_sWv, *py_sWo;
    PyObject *py_n2g, *py_n2b;
    PyObject *py_cWq, *py_cWk, *py_cWv, *py_cWo;
    PyObject *py_n3g, *py_n3b;
    PyObject *py_W1, *py_b1, *py_W2, *py_b2;
    int num_heads;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOOOi",
                          &py_x, &py_enc,
                          &py_n1g, &py_n1b,
                          &py_sWq, &py_sWk, &py_sWv, &py_sWo,
                          &py_n2g, &py_n2b,
                          &py_cWq, &py_cWk, &py_cWv, &py_cWo,
                          &py_n3g, &py_n3b,
                          &py_W1, &py_b1, &py_W2, &py_b2,
                          &num_heads))
        return NULL;

    int tgt_len = (int)PyList_Size(py_x);
    int src_len = (int)PyList_Size(py_enc);
    int d_model = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));
    int d_ff    = (int)PyList_Size(py_b1);

    float *x      = extract_2d(py_x,   tgt_len, d_model);
    float *enc    = extract_2d(py_enc,  src_len, d_model);
    float *n1g    = extract_1d(py_n1g,  d_model);
    float *n1b    = extract_1d(py_n1b,  d_model);
    float *sW_q   = extract_2d(py_sWq,  d_model, d_model);
    float *sW_k   = extract_2d(py_sWk,  d_model, d_model);
    float *sW_v   = extract_2d(py_sWv,  d_model, d_model);
    float *sW_o   = extract_2d(py_sWo,  d_model, d_model);
    float *n2g    = extract_1d(py_n2g,  d_model);
    float *n2b    = extract_1d(py_n2b,  d_model);
    float *cW_q   = extract_2d(py_cWq,  d_model, d_model);
    float *cW_k   = extract_2d(py_cWk,  d_model, d_model);
    float *cW_v   = extract_2d(py_cWv,  d_model, d_model);
    float *cW_o   = extract_2d(py_cWo,  d_model, d_model);
    float *n3g    = extract_1d(py_n3g,  d_model);
    float *n3b    = extract_1d(py_n3b,  d_model);
    float *W1     = extract_2d(py_W1,   d_model, d_ff);
    float *b1     = extract_1d(py_b1,   d_ff);
    float *W2     = extract_2d(py_W2,   d_ff,    d_model);
    float *b2     = extract_1d(py_b2,   d_model);
    float *out    = NULL;
    PyObject *result = NULL;

    if (!x || !enc || !n1g || !n1b ||
        !sW_q || !sW_k || !sW_v || !sW_o ||
        !n2g || !n2b ||
        !cW_q || !cW_k || !cW_v || !cW_o ||
        !n3g || !n3b || !W1 || !b1 || !W2 || !b2) goto cleanup;

    out = (float *)malloc((size_t)tgt_len * d_model * sizeof(float));
    if (!out) { PyErr_NoMemory(); goto cleanup; }

    decoder_layer(out, x, enc,
                  n1g, n1b, sW_q, sW_k, sW_v, sW_o,
                  n2g, n2b, cW_q, cW_k, cW_v, cW_o,
                  n3g, n3b, W1, b1, W2, b2,
                  tgt_len, src_len, d_model, num_heads, d_ff);
    result = pack_2d(out, tgt_len, d_model);

cleanup:
    free(x); free(enc);
    free(n1g); free(n1b);
    free(sW_q); free(sW_k); free(sW_v); free(sW_o);
    free(n2g); free(n2b);
    free(cW_q); free(cW_k); free(cW_v); free(cW_o);
    free(n3g); free(n3b);
    free(W1); free(b1); free(W2); free(b2);
    free(out);
    return result;
}

/* ===========================================================================
 * positional_encoding(max_seq_len, d_model) -> [[float]]
 * =========================================================================*/
static PyObject *py_positional_encoding(PyObject *self, PyObject *args)
{
    int max_seq_len, d_model;
    if (!PyArg_ParseTuple(args, "ii", &max_seq_len, &d_model))
        return NULL;

    float *pe = (float *)malloc((size_t)max_seq_len * d_model * sizeof(float));
    if (!pe) return PyErr_NoMemory();

    positional_encoding(pe, max_seq_len, d_model);
    PyObject *result = pack_2d(pe, max_seq_len, d_model);
    free(pe);
    return result;
}

/* ===========================================================================
 * Helpers for extracting per-layer weight tuples
 * =========================================================================*/

/* Encoder layer: tuple of 12 items (indices 0-11 as documented above). */
typedef struct {
    float *n1g, *n1b;
    float *W_q, *W_k, *W_v, *W_o;
    float *n2g, *n2b;
    float *W1, *b1, *W2, *b2;
} EncLayerWeights;

static void enc_weights_free(EncLayerWeights *w)
{
    free(w->n1g); free(w->n1b);
    free(w->W_q); free(w->W_k); free(w->W_v); free(w->W_o);
    free(w->n2g); free(w->n2b);
    free(w->W1); free(w->b1); free(w->W2); free(w->b2);
}

static int enc_weights_extract(EncLayerWeights *w, PyObject *tup,
                                int d_model, int d_ff)
{
    if (!PySequence_Check(tup) || PySequence_Size(tup) != 12) {
        PyErr_SetString(PyExc_ValueError,
            "each encoder layer weight must be a sequence of 12 items: "
            "(norm1_gamma, norm1_beta, W_q, W_k, W_v, W_o, "
            "norm2_gamma, norm2_beta, W1, b1, W2, b2)");
        return 0;
    }
    PyObject *items[12];
    for (int i = 0; i < 12; i++) {
        items[i] = PySequence_GetItem(tup, i);
        if (!items[i]) return 0;
    }

    w->n1g = extract_1d(items[0],  d_model);
    w->n1b = extract_1d(items[1],  d_model);
    w->W_q = extract_2d(items[2],  d_model, d_model);
    w->W_k = extract_2d(items[3],  d_model, d_model);
    w->W_v = extract_2d(items[4],  d_model, d_model);
    w->W_o = extract_2d(items[5],  d_model, d_model);
    w->n2g = extract_1d(items[6],  d_model);
    w->n2b = extract_1d(items[7],  d_model);
    w->W1  = extract_2d(items[8],  d_model, d_ff);
    w->b1  = extract_1d(items[9],  d_ff);
    w->W2  = extract_2d(items[10], d_ff,    d_model);
    w->b2  = extract_1d(items[11], d_model);

    for (int i = 0; i < 12; i++) Py_DECREF(items[i]);

    return (w->n1g && w->n1b && w->W_q && w->W_k && w->W_v && w->W_o &&
            w->n2g && w->n2b && w->W1  && w->b1  && w->W2  && w->b2);
}

/* Decoder layer: tuple of 18 items (indices 0-17 as documented above). */
typedef struct {
    float *n1g, *n1b;
    float *sW_q, *sW_k, *sW_v, *sW_o;
    float *n2g, *n2b;
    float *cW_q, *cW_k, *cW_v, *cW_o;
    float *n3g, *n3b;
    float *W1, *b1, *W2, *b2;
} DecLayerWeights;

static void dec_weights_free(DecLayerWeights *w)
{
    free(w->n1g); free(w->n1b);
    free(w->sW_q); free(w->sW_k); free(w->sW_v); free(w->sW_o);
    free(w->n2g); free(w->n2b);
    free(w->cW_q); free(w->cW_k); free(w->cW_v); free(w->cW_o);
    free(w->n3g); free(w->n3b);
    free(w->W1); free(w->b1); free(w->W2); free(w->b2);
}

static int dec_weights_extract(DecLayerWeights *w, PyObject *tup,
                                int d_model, int d_ff)
{
    if (!PySequence_Check(tup) || PySequence_Size(tup) != 18) {
        PyErr_SetString(PyExc_ValueError,
            "each decoder layer weight must be a sequence of 18 items: "
            "(norm1_gamma, norm1_beta, "
            "self_W_q, self_W_k, self_W_v, self_W_o, "
            "norm2_gamma, norm2_beta, "
            "cross_W_q, cross_W_k, cross_W_v, cross_W_o, "
            "norm3_gamma, norm3_beta, "
            "W1, b1, W2, b2)");
        return 0;
    }
    PyObject *items[18];
    for (int i = 0; i < 18; i++) {
        items[i] = PySequence_GetItem(tup, i);
        if (!items[i]) return 0;
    }

    w->n1g  = extract_1d(items[0],  d_model);
    w->n1b  = extract_1d(items[1],  d_model);
    w->sW_q = extract_2d(items[2],  d_model, d_model);
    w->sW_k = extract_2d(items[3],  d_model, d_model);
    w->sW_v = extract_2d(items[4],  d_model, d_model);
    w->sW_o = extract_2d(items[5],  d_model, d_model);
    w->n2g  = extract_1d(items[6],  d_model);
    w->n2b  = extract_1d(items[7],  d_model);
    w->cW_q = extract_2d(items[8],  d_model, d_model);
    w->cW_k = extract_2d(items[9],  d_model, d_model);
    w->cW_v = extract_2d(items[10], d_model, d_model);
    w->cW_o = extract_2d(items[11], d_model, d_model);
    w->n3g  = extract_1d(items[12], d_model);
    w->n3b  = extract_1d(items[13], d_model);
    w->W1   = extract_2d(items[14], d_model, d_ff);
    w->b1   = extract_1d(items[15], d_ff);
    w->W2   = extract_2d(items[16], d_ff,    d_model);
    w->b2   = extract_1d(items[17], d_model);

    for (int i = 0; i < 18; i++) Py_DECREF(items[i]);

    return (w->n1g && w->n1b  &&
            w->sW_q && w->sW_k && w->sW_v && w->sW_o &&
            w->n2g && w->n2b  &&
            w->cW_q && w->cW_k && w->cW_v && w->cW_o &&
            w->n3g && w->n3b  &&
            w->W1 && w->b1 && w->W2 && w->b2);
}

/* ===========================================================================
 * encoder_forward(x, layer_weights, num_heads) -> [[float]]
 *
 * layer_weights: Python list of N 12-tuples (one per encoder layer).
 * =========================================================================*/
static PyObject *py_encoder_forward(PyObject *self, PyObject *args)
{
    PyObject *py_x, *py_layers;
    int num_heads;

    if (!PyArg_ParseTuple(args, "OOi", &py_x, &py_layers, &num_heads))
        return NULL;

    if (!PyList_Check(py_x) || PyList_Size(py_x) == 0) {
        PyErr_SetString(PyExc_ValueError, "x must be a non-empty list");
        return NULL;
    }
    if (!PyList_Check(py_layers)) {
        PyErr_SetString(PyExc_ValueError, "layer_weights must be a list");
        return NULL;
    }

    int seq_len   = (int)PyList_Size(py_x);
    int d_model   = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));
    int num_layers = (int)PyList_Size(py_layers);

    if (d_model % num_heads != 0) {
        PyErr_SetString(PyExc_ValueError, "d_model must be divisible by num_heads");
        return NULL;
    }

    /* We need to know d_ff to extract W1/W2.  Infer from the first layer's b1
     * (item index 9 in the tuple), which has length d_ff. */
    int d_ff = 0;
    if (num_layers > 0) {
        PyObject *first = PyList_GET_ITEM(py_layers, 0);
        PyObject *b1_item = PySequence_GetItem(first, 9);
        if (!b1_item) return NULL;
        d_ff = (int)PySequence_Size(b1_item);
        Py_DECREF(b1_item);
    }

    float *cur  = (float *)malloc((size_t)seq_len * d_model * sizeof(float));
    float *next = (float *)malloc((size_t)seq_len * d_model * sizeof(float));
    if (!cur || !next) {
        free(cur); free(next);
        return PyErr_NoMemory();
    }

    /* Extract input */
    float *x_tmp = extract_2d(py_x, seq_len, d_model);
    if (!x_tmp) { free(cur); free(next); return NULL; }
    memcpy(cur, x_tmp, (size_t)seq_len * d_model * sizeof(float));
    free(x_tmp);

    /* Pass through each encoder layer */
    for (int l = 0; l < num_layers; l++) {
        PyObject *layer_tup = PyList_GET_ITEM(py_layers, l);
        EncLayerWeights w;
        memset(&w, 0, sizeof(w));
        if (!enc_weights_extract(&w, layer_tup, d_model, d_ff)) {
            enc_weights_free(&w);
            free(cur); free(next);
            return NULL;
        }

        encoder_layer(next, cur,
                      w.n1g, w.n1b, w.W_q, w.W_k, w.W_v, w.W_o,
                      w.n2g, w.n2b, w.W1, w.b1, w.W2, w.b2,
                      seq_len, d_model, num_heads, d_ff);

        enc_weights_free(&w);

        /* Swap cur and next */
        float *tmp = cur; cur = next; next = tmp;
    }

    PyObject *result = pack_2d(cur, seq_len, d_model);
    free(cur); free(next);
    return result;
}

/* ===========================================================================
 * decoder_forward(x, enc_out, layer_weights, num_heads) -> [[float]]
 *
 * layer_weights: Python list of N 18-tuples (one per decoder layer).
 * =========================================================================*/
static PyObject *py_decoder_forward(PyObject *self, PyObject *args)
{
    PyObject *py_x, *py_enc, *py_layers;
    int num_heads;

    if (!PyArg_ParseTuple(args, "OOOi", &py_x, &py_enc, &py_layers, &num_heads))
        return NULL;

    if (!PyList_Check(py_x) || PyList_Size(py_x) == 0) {
        PyErr_SetString(PyExc_ValueError, "x must be a non-empty list");
        return NULL;
    }
    if (!PyList_Check(py_enc) || PyList_Size(py_enc) == 0) {
        PyErr_SetString(PyExc_ValueError, "enc_out must be a non-empty list");
        return NULL;
    }
    if (!PyList_Check(py_layers)) {
        PyErr_SetString(PyExc_ValueError, "layer_weights must be a list");
        return NULL;
    }

    int tgt_len    = (int)PyList_Size(py_x);
    int src_len    = (int)PyList_Size(py_enc);
    int d_model    = (int)PyList_Size(PyList_GET_ITEM(py_x, 0));
    int num_layers = (int)PyList_Size(py_layers);

    if (d_model % num_heads != 0) {
        PyErr_SetString(PyExc_ValueError, "d_model must be divisible by num_heads");
        return NULL;
    }

    int d_ff = 0;
    if (num_layers > 0) {
        PyObject *first   = PyList_GET_ITEM(py_layers, 0);
        PyObject *b1_item = PySequence_GetItem(first, 15); /* b1 is at index 15 */
        if (!b1_item) return NULL;
        d_ff = (int)PySequence_Size(b1_item);
        Py_DECREF(b1_item);
    }

    /* Extract encoder output (constant across layers) */
    float *enc = extract_2d(py_enc, src_len, d_model);
    if (!enc) return NULL;

    float *cur  = (float *)malloc((size_t)tgt_len * d_model * sizeof(float));
    float *next = (float *)malloc((size_t)tgt_len * d_model * sizeof(float));
    if (!cur || !next) {
        free(enc); free(cur); free(next);
        return PyErr_NoMemory();
    }

    float *x_tmp = extract_2d(py_x, tgt_len, d_model);
    if (!x_tmp) { free(enc); free(cur); free(next); return NULL; }
    memcpy(cur, x_tmp, (size_t)tgt_len * d_model * sizeof(float));
    free(x_tmp);

    for (int l = 0; l < num_layers; l++) {
        PyObject *layer_tup = PyList_GET_ITEM(py_layers, l);
        DecLayerWeights w;
        memset(&w, 0, sizeof(w));
        if (!dec_weights_extract(&w, layer_tup, d_model, d_ff)) {
            dec_weights_free(&w);
            free(enc); free(cur); free(next);
            return NULL;
        }

        decoder_layer(next, cur, enc,
                      w.n1g, w.n1b, w.sW_q, w.sW_k, w.sW_v, w.sW_o,
                      w.n2g, w.n2b, w.cW_q, w.cW_k, w.cW_v, w.cW_o,
                      w.n3g, w.n3b, w.W1, w.b1, w.W2, w.b2,
                      tgt_len, src_len, d_model, num_heads, d_ff);

        dec_weights_free(&w);

        float *tmp = cur; cur = next; next = tmp;
    }

    PyObject *result = pack_2d(cur, tgt_len, d_model);
    free(enc); free(cur); free(next);
    return result;
}

/* ===========================================================================
 * Module definition
 * =========================================================================*/
static PyMethodDef TransformerMethods[] = {
    /* Component-level */
    {"layer_norm",
     py_layer_norm, METH_VARARGS,
     "layer_norm(x, gamma, beta) -> [[float]]\n\n"
     "Pre-LN layer normalization applied to each token in x.\n"
     "x: [seq_len][d_model], gamma/beta: [d_model]"},

    {"multi_head_attention",
     py_multi_head_attention, METH_VARARGS,
     "multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads,\n"
     "                     causal_mask=0) -> [[float]]\n\n"
     "Scaled dot-product multi-head attention.\n"
     "Q: [seq_q][d_model], K/V: [seq_k][d_model], weight matrices: [d_model][d_model]\n"
     "Set causal_mask=1 for decoder masked self-attention."},

    {"ffn_forward",
     py_ffn_forward, METH_VARARGS,
     "ffn_forward(x, W1, b1, W2, b2) -> [[float]]\n\n"
     "Position-wise FFN: ReLU(x @ W1 + b1) @ W2 + b2.\n"
     "x: [seq_len][d_model], W1: [d_model][d_ff], W2: [d_ff][d_model]"},

    {"encoder_layer",
     py_encoder_layer, METH_VARARGS,
     "encoder_layer(x, norm1_gamma, norm1_beta,\n"
     "              W_q, W_k, W_v, W_o,\n"
     "              norm2_gamma, norm2_beta,\n"
     "              W1, b1, W2, b2, num_heads) -> [[float]]\n\n"
     "Single Pre-LN encoder layer (self-attention + FFN with residuals)."},

    {"decoder_layer",
     py_decoder_layer, METH_VARARGS,
     "decoder_layer(x, enc_out,\n"
     "              norm1_gamma, norm1_beta,\n"
     "              self_W_q, self_W_k, self_W_v, self_W_o,\n"
     "              norm2_gamma, norm2_beta,\n"
     "              cross_W_q, cross_W_k, cross_W_v, cross_W_o,\n"
     "              norm3_gamma, norm3_beta,\n"
     "              W1, b1, W2, b2, num_heads) -> [[float]]\n\n"
     "Single Pre-LN decoder layer (masked self-attn + cross-attn + FFN)."},

    {"positional_encoding",
     py_positional_encoding, METH_VARARGS,
     "positional_encoding(max_seq_len, d_model) -> [[float]]\n\n"
     "Sinusoidal positional encoding table of shape [max_seq_len][d_model]."},

    /* Stack-level */
    {"encoder_forward",
     py_encoder_forward, METH_VARARGS,
     "encoder_forward(x, layer_weights, num_heads) -> [[float]]\n\n"
     "Run a full Pre-LN encoder stack.\n"
     "x: [seq_len][d_model]\n"
     "layer_weights: list of 12-tuples, one per layer:\n"
     "  (norm1_gamma, norm1_beta, W_q, W_k, W_v, W_o,\n"
     "   norm2_gamma, norm2_beta, W1, b1, W2, b2)"},

    {"decoder_forward",
     py_decoder_forward, METH_VARARGS,
     "decoder_forward(x, enc_out, layer_weights, num_heads) -> [[float]]\n\n"
     "Run a full Pre-LN decoder stack.\n"
     "x: [tgt_seq_len][d_model], enc_out: [src_seq_len][d_model]\n"
     "layer_weights: list of 18-tuples, one per layer:\n"
     "  (norm1_gamma, norm1_beta,\n"
     "   self_W_q, self_W_k, self_W_v, self_W_o,\n"
     "   norm2_gamma, norm2_beta,\n"
     "   cross_W_q, cross_W_k, cross_W_v, cross_W_o,\n"
     "   norm3_gamma, norm3_beta,\n"
     "   W1, b1, W2, b2)"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef transformermodule = {
    PyModuleDef_HEAD_INIT,
    "transformer_module",
    "Pre-LN Transformer (encoder + decoder) implemented in C.\n\n"
    "Architecture: 'Attention Is All You Need' (Vaswani et al., 2017)\n"
    "with Pre-Layer Normalization for improved training stability.\n\n"
    "Use help(transformer_module.<function>) for detailed signatures.",
    -1,
    TransformerMethods
};

PyMODINIT_FUNC PyInit_transformer_module(void)
{
    return PyModule_Create(&transformermodule);
}
