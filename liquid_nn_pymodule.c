/*
 * Python C extension for LiquidNN
 *
 * Exposes:
 *   lnn.LiquidNN(input_size, reservoir_size, output_size,
 *                dt=0.1, ode_steps=5)
 *     .forward(input: list[float]) -> list[float]
 *     .train_step(input: list[float], target: list[float],
 *                 lr: float = 0.001) -> float
 *     .reset_state()
 *     .save(filename: str) -> int
 *     LiquidNN.load(filename: str) -> LiquidNN
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "liquid_nn.h"

/* ── Python object wrapper ───────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    LiquidNN *net;
} LiquidNNObject;

/* ── helpers ─────────────────────────────────────────────────── */

/* Convert a Python sequence to a freshly-malloc'd float array.
   Returns NULL and sets a Python exception on error. */
static float *seq_to_floats(PyObject *seq, int expected_len) {
    Py_ssize_t n = PySequence_Length(seq);
    if (n < 0) return NULL;
    if (expected_len >= 0 && n != (Py_ssize_t)expected_len) {
        PyErr_Format(PyExc_ValueError,
                     "expected sequence of length %d, got %zd",
                     expected_len, n);
        return NULL;
    }
    float *arr = (float *)malloc((size_t)n * sizeof(float));
    if (!arr) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (!item) { free(arr); return NULL; }
        arr[i] = (float)PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) { free(arr); return NULL; }
    }
    return arr;
}

/* Convert a float array of length n to a Python list of floats. */
static PyObject *floats_to_list(const float *arr, int n) {
    PyObject *lst = PyList_New(n);
    if (!lst) return NULL;
    for (int i = 0; i < n; i++)
        PyList_SET_ITEM(lst, i, PyFloat_FromDouble((double)arr[i]));
    return lst;
}

/* ── tp_new / tp_init / tp_dealloc ───────────────────────────── */

static PyObject *LiquidNN_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds) {
    LiquidNNObject *self = (LiquidNNObject *)type->tp_alloc(type, 0);
    if (self) self->net = NULL;
    return (PyObject *)self;
}

static int LiquidNN_init(LiquidNNObject *self, PyObject *args, PyObject *kwds) {
    int   input_size, reservoir_size, output_size;
    float dt        = 0.1f;
    int   ode_steps = 5;

    static char *kwlist[] = {
        "input_size", "reservoir_size", "output_size",
        "dt", "ode_steps", NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii|fi", kwlist,
                                     &input_size, &reservoir_size, &output_size,
                                     &dt, &ode_steps))
        return -1;

    if (input_size <= 0 || reservoir_size <= 0 || output_size <= 0 ||
        dt <= 0.0f || ode_steps <= 0) {
        PyErr_SetString(PyExc_ValueError, "all dimensions and dt must be > 0");
        return -1;
    }

    lnn_free(self->net);
    self->net = lnn_create(input_size, reservoir_size, output_size,
                           dt, ode_steps);
    if (!self->net) { PyErr_NoMemory(); return -1; }
    return 0;
}

static void LiquidNN_dealloc(LiquidNNObject *self) {
    lnn_free(self->net);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ── methods ──────────────────────────────────────────────────── */

static PyObject *LiquidNN_forward(LiquidNNObject *self, PyObject *args) {
    PyObject *input_seq;
    if (!PyArg_ParseTuple(args, "O", &input_seq)) return NULL;
    if (!self->net) { PyErr_SetString(PyExc_RuntimeError, "not initialised"); return NULL; }

    float *input = seq_to_floats(input_seq, self->net->input_size);
    if (!input) return NULL;

    float *output = (float *)malloc((size_t)self->net->output_size * sizeof(float));
    if (!output) { free(input); PyErr_NoMemory(); return NULL; }

    lnn_forward(self->net, input, output);

    PyObject *result = floats_to_list(output, self->net->output_size);
    free(input);
    free(output);
    return result;
}

static PyObject *LiquidNN_train_step(LiquidNNObject *self, PyObject *args,
                                     PyObject *kwds) {
    PyObject *input_seq, *target_seq;
    float lr = 0.001f;

    static char *kwlist[] = {"input", "target", "lr", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|f", kwlist,
                                     &input_seq, &target_seq, &lr))
        return NULL;
    if (!self->net) { PyErr_SetString(PyExc_RuntimeError, "not initialised"); return NULL; }

    float *input  = seq_to_floats(input_seq,  self->net->input_size);
    if (!input) return NULL;
    float *target = seq_to_floats(target_seq, self->net->output_size);
    if (!target) { free(input); return NULL; }

    float loss = lnn_train_step(self->net, input, target, lr);

    free(input);
    free(target);
    return PyFloat_FromDouble((double)loss);
}

static PyObject *LiquidNN_reset_state(LiquidNNObject *self, PyObject *args) {
    if (!self->net) { PyErr_SetString(PyExc_RuntimeError, "not initialised"); return NULL; }
    lnn_reset_state(self->net);
    Py_RETURN_NONE;
}

static PyObject *LiquidNN_save(LiquidNNObject *self, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;
    if (!self->net) { PyErr_SetString(PyExc_RuntimeError, "not initialised"); return NULL; }
    int rc = lnn_save(self->net, filename);
    if (rc != 0) { PyErr_SetString(PyExc_IOError, "save failed"); return NULL; }
    Py_RETURN_NONE;
}

/* classmethod: LiquidNN.load(filename) -> LiquidNN */
static PyObject *LiquidNN_load(PyObject *cls, PyObject *args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;

    LiquidNN *net = lnn_load(filename);
    if (!net) { PyErr_SetString(PyExc_IOError, "load failed or bad file"); return NULL; }

    LiquidNNObject *obj = (LiquidNNObject *)
        ((PyTypeObject *)cls)->tp_alloc((PyTypeObject *)cls, 0);
    if (!obj) { lnn_free(net); return NULL; }
    obj->net = net;
    return (PyObject *)obj;
}

/* properties */
static PyObject *LiquidNN_get_input_size(LiquidNNObject *self, void *closure) {
    if (!self->net) Py_RETURN_NONE;
    return PyLong_FromLong(self->net->input_size);
}
static PyObject *LiquidNN_get_reservoir_size(LiquidNNObject *self, void *closure) {
    if (!self->net) Py_RETURN_NONE;
    return PyLong_FromLong(self->net->reservoir_size);
}
static PyObject *LiquidNN_get_output_size(LiquidNNObject *self, void *closure) {
    if (!self->net) Py_RETURN_NONE;
    return PyLong_FromLong(self->net->output_size);
}
static PyObject *LiquidNN_get_state(LiquidNNObject *self, void *closure) {
    if (!self->net) Py_RETURN_NONE;
    return floats_to_list(self->net->state, self->net->reservoir_size);
}

static PyGetSetDef LiquidNN_getset[] = {
    {"input_size",     (getter)LiquidNN_get_input_size,     NULL, "input size",     NULL},
    {"reservoir_size", (getter)LiquidNN_get_reservoir_size, NULL, "reservoir size", NULL},
    {"output_size",    (getter)LiquidNN_get_output_size,    NULL, "output size",    NULL},
    {"state",          (getter)LiquidNN_get_state,          NULL, "current hidden state", NULL},
    {NULL}
};

static PyMethodDef LiquidNN_methods[] = {
    {"forward",     (PyCFunction)LiquidNN_forward,     METH_VARARGS,
     "forward(input) -> list[float]\nRun forward pass."},
    {"train_step",  (PyCFunction)LiquidNN_train_step,  METH_VARARGS | METH_KEYWORDS,
     "train_step(input, target, lr=0.001) -> float\nOne BPTT step, returns MSE loss."},
    {"reset_state", (PyCFunction)LiquidNN_reset_state, METH_NOARGS,
     "reset_state()\nZero the hidden state."},
    {"save",        (PyCFunction)LiquidNN_save,        METH_VARARGS,
     "save(filename)\nSerialise model to binary file."},
    {"load",        (PyCFunction)LiquidNN_load,        METH_CLASS | METH_VARARGS,
     "load(filename) -> LiquidNN\nDeserialise model from binary file."},
    {NULL}
};

static PyTypeObject LiquidNNType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "lnn.LiquidNN",
    .tp_basicsize = sizeof(LiquidNNObject),
    .tp_dealloc   = (destructor)LiquidNN_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Liquid Neural Network (Continuous-Time RNN with BPTT).",
    .tp_methods   = LiquidNN_methods,
    .tp_getset    = LiquidNN_getset,
    .tp_init      = (initproc)LiquidNN_init,
    .tp_new       = LiquidNN_new,
};

/* ── module ───────────────────────────────────────────────────── */

static PyModuleDef lnn_module = {
    PyModuleDef_HEAD_INIT, "lnn",
    "Liquid Neural Network C extension.", -1, NULL
};

PyMODINIT_FUNC PyInit_lnn(void) {
    if (PyType_Ready(&LiquidNNType) < 0) return NULL;

    PyObject *m = PyModule_Create(&lnn_module);
    if (!m) return NULL;

    Py_INCREF(&LiquidNNType);
    if (PyModule_AddObject(m, "LiquidNN", (PyObject *)&LiquidNNType) < 0) {
        Py_DECREF(&LiquidNNType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
