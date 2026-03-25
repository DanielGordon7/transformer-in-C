#ifndef LIQUID_NN_H
#define LIQUID_NN_H

/*
 * Liquid Neural Network (Continuous-Time RNN variant)
 *
 * ODE:  tau * dx/dt = -x + tanh(W_in*u + W_rec*x + b_rec)
 * Euler integration with configurable step size and number of steps.
 * Output: y = W_out * x + b_out
 */

typedef struct {
    int input_size;
    int reservoir_size;
    int output_size;

    float *W_in;   /* [reservoir_size x input_size]  */
    float *W_rec;  /* [reservoir_size x reservoir_size] */
    float *W_out;  /* [output_size    x reservoir_size] */
    float *b_rec;  /* [reservoir_size] */
    float *b_out;  /* [output_size]    */
    float *tau;    /* [reservoir_size] time constants  */
    float *state;  /* [reservoir_size] current hidden state */

    float dt;       /* Euler integration step size */
    int   ode_steps;/* Number of ODE integration steps per input */
} LiquidNN;

/* Lifecycle */
LiquidNN *lnn_create(int input_size, int reservoir_size, int output_size,
                     float dt, int ode_steps);
void      lnn_free(LiquidNN *lnn);
void      lnn_reset_state(LiquidNN *lnn);

/* Inference */
void lnn_forward(LiquidNN *lnn, const float *input, float *output);

/* Training: returns MSE loss, updates all weights via BPTT */
float lnn_train_step(LiquidNN *lnn, const float *input, const float *target,
                     float lr);

/* Persistence */
int       lnn_save(const LiquidNN *lnn, const char *filename);
LiquidNN *lnn_load(const char *filename);

#endif /* LIQUID_NN_H */
