#include "liquid_nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── helpers ─────────────────────────────────────────────────── */

static float randf(float lo, float hi) {
    return lo + (float)rand() / (float)RAND_MAX * (hi - lo);
}

/* Clip gradient vector in-place to max L2 norm. */
static void clip_grad(float *g, int n, float max_norm) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += g[i] * g[i];
    norm = sqrtf(norm);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; i++) g[i] *= scale;
    }
}

/* ── lifecycle ───────────────────────────────────────────────── */

LiquidNN *lnn_create(int input_size, int reservoir_size, int output_size,
                     float dt, int ode_steps) {
    LiquidNN *lnn = (LiquidNN *)calloc(1, sizeof(LiquidNN));
    if (!lnn) return NULL;

    lnn->input_size     = input_size;
    lnn->reservoir_size = reservoir_size;
    lnn->output_size    = output_size;
    lnn->dt             = dt;
    lnn->ode_steps      = ode_steps;

    int R = reservoir_size, I = input_size, O = output_size;

    lnn->W_in  = (float *)malloc(R * I * sizeof(float));
    lnn->W_rec = (float *)malloc(R * R * sizeof(float));
    lnn->W_out = (float *)calloc(O * R,   sizeof(float));
    lnn->b_rec = (float *)calloc(R,        sizeof(float));
    lnn->b_out = (float *)calloc(O,        sizeof(float));
    lnn->tau   = (float *)malloc(R *       sizeof(float));
    lnn->state = (float *)calloc(R,        sizeof(float));

    if (!lnn->W_in || !lnn->W_rec || !lnn->W_out ||
        !lnn->b_rec || !lnn->b_out || !lnn->tau || !lnn->state) {
        lnn_free(lnn);
        return NULL;
    }

    srand((unsigned)time(NULL));

    /* W_in: uniform [-0.5, 0.5] */
    for (int i = 0; i < R * I; i++)
        lnn->W_in[i] = randf(-0.5f, 0.5f);

    /* W_rec: ~20 % sparse, scaled so spectral radius ≈ 0.9 */
    float w_scale = 0.9f / sqrtf((float)R * 0.2f + 1e-6f);
    for (int i = 0; i < R * R; i++) {
        lnn->W_rec[i] = ((float)rand() / RAND_MAX < 0.2f)
                            ? randf(-1.0f, 1.0f) * w_scale
                            : 0.0f;
    }

    /* W_out stays zero; learned from scratch */

    /* tau: uniform [0.5, 2.0] */
    for (int i = 0; i < R; i++)
        lnn->tau[i] = randf(0.5f, 2.0f);

    return lnn;
}

void lnn_free(LiquidNN *lnn) {
    if (!lnn) return;
    free(lnn->W_in);
    free(lnn->W_rec);
    free(lnn->W_out);
    free(lnn->b_rec);
    free(lnn->b_out);
    free(lnn->tau);
    free(lnn->state);
    free(lnn);
}

void lnn_reset_state(LiquidNN *lnn) {
    memset(lnn->state, 0, (size_t)lnn->reservoir_size * sizeof(float));
}

/* ── forward pass ────────────────────────────────────────────── */
/*
 * ODE step (Euler):
 *   alpha[r] = dt / tau[r]
 *   x[r] <- x[r]*(1-alpha[r]) + alpha[r]*tanh(W_in*u + W_rec*x + b_rec)[r]
 *
 * Output: y[o] = W_out[o,:] · x + b_out[o]
 */
void lnn_forward(LiquidNN *lnn, const float *input, float *output) {
    int R = lnn->reservoir_size;
    int I = lnn->input_size;
    int O = lnn->output_size;
    float dt = lnn->dt;

    float *x     = lnn->state;
    float *x_new = (float *)malloc((size_t)R * sizeof(float));

    for (int step = 0; step < lnn->ode_steps; step++) {
        for (int r = 0; r < R; r++) {
            float net = lnn->b_rec[r];
            for (int i = 0; i < I; i++)
                net += lnn->W_in[r * I + i] * input[i];
            for (int j = 0; j < R; j++)
                net += lnn->W_rec[r * R + j] * x[j];

            float alpha  = dt / lnn->tau[r];
            x_new[r] = x[r] * (1.0f - alpha) + alpha * tanhf(net);
        }
        memcpy(x, x_new, (size_t)R * sizeof(float));
    }
    free(x_new);

    for (int o = 0; o < O; o++) {
        float val = lnn->b_out[o];
        for (int r = 0; r < R; r++)
            val += lnn->W_out[o * R + r] * x[r];
        output[o] = val;
    }
}

/* ── training step (BPTT through ODE) ───────────────────────── */
/*
 * Returns MSE loss.  Gradients are clipped by global L2 norm (max=5).
 *
 * Notation
 *   states[k]   = x after k ODE steps   (k = 0 … ode_steps)
 *   acts[k]     = tanh(net) at step k   (k = 0 … ode_steps-1)
 *   alpha[r]    = dt / tau[r]
 *
 * Backprop through one ODE step:
 *   e[r]   = alpha[r] * (1 - acts[k][r]^2) * dl_dx[r]
 *   dW_in  += e ⊗ u
 *   dW_rec += e ⊗ x[k]
 *   db_rec += e
 *   dl_dx_prev[j] = (1-alpha[j])*dl_dx[j] + Σ_r e[r]*W_rec[r,j]
 */
float lnn_train_step(LiquidNN *lnn, const float *input, const float *target,
                     float lr) {
    int R  = lnn->reservoir_size;
    int I  = lnn->input_size;
    int O  = lnn->output_size;
    int S  = lnn->ode_steps;
    float dt = lnn->dt;

    /* ---- allocate forward buffers ---- */
    float *states = (float *)malloc((size_t)(S + 1) * R * sizeof(float));
    float *acts   = (float *)malloc((size_t) S      * R * sizeof(float));

    /* initial state */
    memcpy(states, lnn->state, (size_t)R * sizeof(float));

    /* ---- forward pass ---- */
    for (int step = 0; step < S; step++) {
        float *xp = states + step       * R;
        float *xn = states + (step + 1) * R;
        float *f  = acts   + step       * R;

        for (int r = 0; r < R; r++) {
            float net = lnn->b_rec[r];
            for (int i = 0; i < I; i++)
                net += lnn->W_in[r * I + i] * input[i];
            for (int j = 0; j < R; j++)
                net += lnn->W_rec[r * R + j] * xp[j];

            f[r] = tanhf(net);
            float alpha = dt / lnn->tau[r];
            xn[r] = xp[r] * (1.0f - alpha) + alpha * f[r];
        }
    }

    /* update model state */
    memcpy(lnn->state, states + S * R, (size_t)R * sizeof(float));
    float *x_final = lnn->state;

    /* ---- output & loss ---- */
    float *output = (float *)malloc((size_t)O * sizeof(float));
    for (int o = 0; o < O; o++) {
        float val = lnn->b_out[o];
        for (int r = 0; r < R; r++)
            val += lnn->W_out[o * R + r] * x_final[r];
        output[o] = val;
    }

    float loss = 0.0f;
    float *dl_dy = (float *)malloc((size_t)O * sizeof(float));
    for (int o = 0; o < O; o++) {
        float diff = output[o] - target[o];
        loss     += diff * diff;
        dl_dy[o]  = 2.0f * diff / (float)O;
    }
    loss /= (float)O;

    /* ---- gradient buffers ---- */
    float *dW_out = (float *)calloc((size_t)O * R, sizeof(float));
    float *db_out = (float *)calloc((size_t)O,     sizeof(float));
    float *dW_in  = (float *)calloc((size_t)R * I, sizeof(float));
    float *dW_rec = (float *)calloc((size_t)R * R, sizeof(float));
    float *db_rec = (float *)calloc((size_t)R,     sizeof(float));

    /* dl/dx at step S */
    float *dl_dx      = (float *)calloc((size_t)R, sizeof(float));
    float *dl_dx_prev = (float *)calloc((size_t)R, sizeof(float));
    float *e          = (float *)malloc((size_t)R  * sizeof(float));

    /* output layer gradients + seed dl_dx */
    for (int o = 0; o < O; o++) {
        for (int r = 0; r < R; r++) {
            dW_out[o * R + r]  = dl_dy[o] * x_final[r];
            dl_dx[r]          += lnn->W_out[o * R + r] * dl_dy[o];
        }
        db_out[o] = dl_dy[o];
    }

    /* ---- BPTT through ODE steps ---- */
    for (int step = S - 1; step >= 0; step--) {
        float *xp = states + step * R;
        float *f  = acts   + step * R;

        for (int r = 0; r < R; r++) {
            float alpha = dt / lnn->tau[r];
            e[r] = alpha * (1.0f - f[r] * f[r]) * dl_dx[r];
        }

        for (int r = 0; r < R; r++) {
            for (int i = 0; i < I; i++)
                dW_in[r * I + i] += e[r] * input[i];
            for (int j = 0; j < R; j++)
                dW_rec[r * R + j] += e[r] * xp[j];
            db_rec[r] += e[r];
        }

        /* dl_dx_prev[j] = (1-alpha[j])*dl_dx[j] + Σ_r e[r]*W_rec[r,j] */
        for (int j = 0; j < R; j++) {
            float alpha_j = dt / lnn->tau[j];
            float val = (1.0f - alpha_j) * dl_dx[j];
            for (int r = 0; r < R; r++)
                val += e[r] * lnn->W_rec[r * R + j];
            dl_dx_prev[j] = val;
        }

        memcpy(dl_dx, dl_dx_prev, (size_t)R * sizeof(float));
    }

    /* ---- gradient clipping ---- */
    clip_grad(dW_out, O * R, 5.0f);
    clip_grad(db_out, O,     5.0f);
    clip_grad(dW_in,  R * I, 5.0f);
    clip_grad(dW_rec, R * R, 5.0f);
    clip_grad(db_rec, R,     5.0f);

    /* ---- weight update ---- */
    for (int i = 0; i < O * R; i++) lnn->W_out[i] -= lr * dW_out[i];
    for (int i = 0; i < O;     i++) lnn->b_out[i] -= lr * db_out[i];
    for (int i = 0; i < R * I; i++) lnn->W_in[i]  -= lr * dW_in[i];
    for (int i = 0; i < R * R; i++) lnn->W_rec[i]  -= lr * dW_rec[i];
    for (int i = 0; i < R;     i++) lnn->b_rec[i]  -= lr * db_rec[i];

    /* ---- cleanup ---- */
    free(states); free(acts);
    free(output); free(dl_dy);
    free(dW_out); free(db_out);
    free(dW_in);  free(dW_rec); free(db_rec);
    free(dl_dx);  free(dl_dx_prev); free(e);

    return loss;
}

/* ── sequence inference ──────────────────────────────────────── */

void lnn_forward_sequence(LiquidNN *lnn, const float *inputs, int T,
                          float *output) {
    int I = lnn->input_size;
    for (int t = 0; t < T; t++)
        lnn_forward(lnn, inputs + t * I, output);
    /* output already written by the last lnn_forward call */
}

/* ── sequence training (full BPTT) ───────────────────────────── */
/*
 * All T inputs share the same weight matrices.
 * The integration steps are indexed globally: step = t*S + s
 * where t ∈ [0,T) is the input timestep and s ∈ [0,S) is the ODE
 * sub-step within that timestep.
 */
float lnn_train_sequence(LiquidNN *lnn, const float *inputs, int T,
                         const float *target, float lr) {
    int R  = lnn->reservoir_size;
    int I  = lnn->input_size;
    int O  = lnn->output_size;
    int S  = lnn->ode_steps;
    float dt = lnn->dt;
    int total = T * S;

    float *states = (float *)malloc((size_t)(total + 1) * R * sizeof(float));
    float *acts   = (float *)malloc((size_t) total      * R * sizeof(float));

    memcpy(states, lnn->state, (size_t)R * sizeof(float));

    /* ---- forward ---- */
    for (int t = 0; t < T; t++) {
        const float *u = inputs + t * I;
        for (int s = 0; s < S; s++) {
            int step    = t * S + s;
            float *xp   = states + step       * R;
            float *xn   = states + (step + 1) * R;
            float *f    = acts   + step        * R;
            for (int r = 0; r < R; r++) {
                float net = lnn->b_rec[r];
                for (int i = 0; i < I; i++)
                    net += lnn->W_in[r * I + i] * u[i];
                for (int j = 0; j < R; j++)
                    net += lnn->W_rec[r * R + j] * xp[j];
                f[r] = tanhf(net);
                float alpha = dt / lnn->tau[r];
                xn[r] = xp[r] * (1.0f - alpha) + alpha * f[r];
            }
        }
    }
    memcpy(lnn->state, states + total * R, (size_t)R * sizeof(float));
    float *x_final = lnn->state;

    /* ---- output & loss ---- */
    float *output = (float *)malloc((size_t)O * sizeof(float));
    for (int o = 0; o < O; o++) {
        float v = lnn->b_out[o];
        for (int r = 0; r < R; r++)
            v += lnn->W_out[o * R + r] * x_final[r];
        output[o] = v;
    }
    float loss = 0.0f;
    float *dl_dy = (float *)malloc((size_t)O * sizeof(float));
    for (int o = 0; o < O; o++) {
        float d = output[o] - target[o];
        loss += d * d;
        dl_dy[o] = 2.0f * d / (float)O;
    }
    loss /= (float)O;

    /* ---- gradient buffers ---- */
    float *dW_out     = (float *)calloc((size_t)O * R, sizeof(float));
    float *db_out     = (float *)calloc((size_t)O,     sizeof(float));
    float *dW_in      = (float *)calloc((size_t)R * I, sizeof(float));
    float *dW_rec     = (float *)calloc((size_t)R * R, sizeof(float));
    float *db_rec     = (float *)calloc((size_t)R,     sizeof(float));
    float *dl_dx      = (float *)calloc((size_t)R,     sizeof(float));
    float *dl_dx_prev = (float *)calloc((size_t)R,     sizeof(float));
    float *e          = (float *)malloc ((size_t)R *    sizeof(float));

    for (int o = 0; o < O; o++) {
        for (int r = 0; r < R; r++) {
            dW_out[o * R + r]  = dl_dy[o] * x_final[r];
            dl_dx[r]          += lnn->W_out[o * R + r] * dl_dy[o];
        }
        db_out[o] = dl_dy[o];
    }

    /* ---- BPTT ---- */
    for (int step = total - 1; step >= 0; step--) {
        int t         = step / S;
        const float *u = inputs + t * I;
        float *xp     = states + step * R;
        float *f      = acts   + step * R;

        for (int r = 0; r < R; r++) {
            float alpha = dt / lnn->tau[r];
            e[r] = alpha * (1.0f - f[r] * f[r]) * dl_dx[r];
        }
        for (int r = 0; r < R; r++) {
            for (int i = 0; i < I; i++)
                dW_in[r * I + i] += e[r] * u[i];
            for (int j = 0; j < R; j++)
                dW_rec[r * R + j] += e[r] * xp[j];
            db_rec[r] += e[r];
        }
        for (int j = 0; j < R; j++) {
            float alpha_j = dt / lnn->tau[j];
            float v = (1.0f - alpha_j) * dl_dx[j];
            for (int r = 0; r < R; r++)
                v += e[r] * lnn->W_rec[r * R + j];
            dl_dx_prev[j] = v;
        }
        memcpy(dl_dx, dl_dx_prev, (size_t)R * sizeof(float));
    }

    clip_grad(dW_out, O * R, 5.0f);
    clip_grad(db_out, O,     5.0f);
    clip_grad(dW_in,  R * I, 5.0f);
    clip_grad(dW_rec, R * R, 5.0f);
    clip_grad(db_rec, R,     5.0f);

    for (int i = 0; i < O * R; i++) lnn->W_out[i] -= lr * dW_out[i];
    for (int i = 0; i < O;     i++) lnn->b_out[i] -= lr * db_out[i];
    for (int i = 0; i < R * I; i++) lnn->W_in[i]  -= lr * dW_in[i];
    for (int i = 0; i < R * R; i++) lnn->W_rec[i]  -= lr * dW_rec[i];
    for (int i = 0; i < R;     i++) lnn->b_rec[i]  -= lr * db_rec[i];

    free(states); free(acts);
    free(output); free(dl_dy);
    free(dW_out); free(db_out);
    free(dW_in);  free(dW_rec); free(db_rec);
    free(dl_dx);  free(dl_dx_prev); free(e);
    return loss;
}

/* ── persistence ─────────────────────────────────────────────── */

#define LNN_MAGIC   0x4C4E4E31u   /* "LNN1" */

int lnn_save(const LiquidNN *lnn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return -1;

    unsigned magic = LNN_MAGIC;
    fwrite(&magic,            sizeof(unsigned), 1, fp);
    fwrite(&lnn->input_size,     sizeof(int),   1, fp);
    fwrite(&lnn->reservoir_size, sizeof(int),   1, fp);
    fwrite(&lnn->output_size,    sizeof(int),   1, fp);
    fwrite(&lnn->dt,             sizeof(float), 1, fp);
    fwrite(&lnn->ode_steps,      sizeof(int),   1, fp);

    int R = lnn->reservoir_size, I = lnn->input_size, O = lnn->output_size;
    fwrite(lnn->W_in,  sizeof(float), (size_t)R * I, fp);
    fwrite(lnn->W_rec, sizeof(float), (size_t)R * R, fp);
    fwrite(lnn->W_out, sizeof(float), (size_t)O * R, fp);
    fwrite(lnn->b_rec, sizeof(float), (size_t)R,     fp);
    fwrite(lnn->b_out, sizeof(float), (size_t)O,     fp);
    fwrite(lnn->tau,   sizeof(float), (size_t)R,     fp);
    fwrite(lnn->state, sizeof(float), (size_t)R,     fp);

    fclose(fp);
    return 0;
}

LiquidNN *lnn_load(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;

    unsigned magic = 0;
#define FREAD(ptr, size, n, fp) \
    do { if (fread((ptr), (size), (n), (fp)) != (size_t)(n)) { fclose(fp); return NULL; } } while(0)

    FREAD(&magic, sizeof(unsigned), 1, fp);
    if (magic != LNN_MAGIC) { fclose(fp); return NULL; }

    int input_size, reservoir_size, output_size, ode_steps;
    float dt;
    FREAD(&input_size,     sizeof(int),   1, fp);
    FREAD(&reservoir_size, sizeof(int),   1, fp);
    FREAD(&output_size,    sizeof(int),   1, fp);
    FREAD(&dt,             sizeof(float), 1, fp);
    FREAD(&ode_steps,      sizeof(int),   1, fp);

    LiquidNN *lnn = lnn_create(input_size, reservoir_size, output_size,
                               dt, ode_steps);
    if (!lnn) { fclose(fp); return NULL; }

    int R = reservoir_size, I = input_size, O = output_size;
    FREAD(lnn->W_in,  sizeof(float), (size_t)R * I, fp);
    FREAD(lnn->W_rec, sizeof(float), (size_t)R * R, fp);
    FREAD(lnn->W_out, sizeof(float), (size_t)O * R, fp);
    FREAD(lnn->b_rec, sizeof(float), (size_t)R,     fp);
    FREAD(lnn->b_out, sizeof(float), (size_t)O,     fp);
    FREAD(lnn->tau,   sizeof(float), (size_t)R,     fp);
    FREAD(lnn->state, sizeof(float), (size_t)R,     fp);

    fclose(fp);
    return lnn;
}
