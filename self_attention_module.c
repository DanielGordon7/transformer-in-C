#define PY_SSIZE_T_CLEAN
#include <Python.h>  // Python/C API
#include "self_attention.h"
#include <stdlib.h>

// Python-callable wrapper: self_attention(Q, K, V) -> list of lists
// every function inputs PyObject *self and returns PyObject *
// every python object is treated as a pointer to PyObject in Python/C API
static PyObject *py_self_attention(PyObject *self, PyObject *args) {
    // input to function will be python lists of lists
    PyObject *Q_list, *K_list, *V_list;

    // parse input arguments: "000" means it requires 3 positional arguments
    // we take their addresses to get the python object themselves
    if (!PyArg_ParseTuple(args, "OOO", &Q_list, &K_list, &V_list))
        return NULL;

    // get dimensions from input list of lists
    int seq_len = (int)PyList_Size(Q_list);
    int d_model = (int)PyList_Size(PyList_GetItem(Q_list, 0));

    float *Q   = (float *)malloc(seq_len * d_model * sizeof(float));
    float *K   = (float *)malloc(seq_len * d_model * sizeof(float));
    float *V   = (float *)malloc(seq_len * d_model * sizeof(float));
    float *out = (float *)malloc(seq_len * d_model * sizeof(float));

    // Unpack Python lists into flat C arrays
    for (int i = 0; i < seq_len; i++) {
        PyObject *q_row = PyList_GetItem(Q_list, i);
        PyObject *k_row = PyList_GetItem(K_list, i);
        PyObject *v_row = PyList_GetItem(V_list, i);
        for (int j = 0; j < d_model; j++) {
            Q[i * d_model + j] = (float)PyFloat_AsDouble(PyList_GetItem(q_row, j));
            K[i * d_model + j] = (float)PyFloat_AsDouble(PyList_GetItem(k_row, j));
            V[i * d_model + j] = (float)PyFloat_AsDouble(PyList_GetItem(v_row, j));
        }
    }

    // call function from self_attention.c, it modifies out in place
    self_attention(Q, K, V, out, seq_len, d_model);

    // Pack C array back into a Python list of lists
    PyObject *result = PyList_New(seq_len);
    for (int i = 0; i < seq_len; i++) {
        PyObject *row = PyList_New(d_model);
        for (int j = 0; j < d_model; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble((double)out[i * d_model + j]));
        }
        PyList_SetItem(result, i, row);  // add row to result list
    }

    free(Q); free(K); free(V); free(out);
    return result;
}

// array of function definitions: "self_attention" is the function name in python
// python interpreted will call py_self_attention when "self_attention" is called
// METH_VARARGS means it takes positional arguments (for keyword arguments use METH_KEYWORDS)
static PyMethodDef AttentionMethods[] = {
    {"self_attention", py_self_attention, METH_VARARGS,
     "Compute scaled dot-product self-attention.\n\n"
     "Args:\n"
     "  Q, K, V: list of lists of floats, shape [seq_len][d_model]\n"
     "Returns:\n"
     "  Output matrix as list of lists, shape [seq_len][d_model]"},
    {NULL, NULL, 0, NULL}
};

// module definition: "self_attention_module" is the module name in python
static struct PyModuleDef attentionmodule = {
    PyModuleDef_HEAD_INIT, "self_attention_module", NULL, -1, AttentionMethods
};

// this function is called on 'import self_attention_module' in python
// name of function need to be 'PyInit_<module_name>'
PyMODINIT_FUNC PyInit_self_attention_module(void) {
    return PyModule_Create(&attentionmodule);
}
