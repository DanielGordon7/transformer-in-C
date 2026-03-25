from setuptools import setup, Extension

lnn_ext = Extension(
    name="lnn",
    sources=["liquid_nn.c", "liquid_nn_pymodule.c"],
    extra_compile_args=["-O2", "-Wall", "-std=c11"],
    extra_link_args=["-lm"],
)

setup(
    name="lnn",
    version="0.1.0",
    description="Liquid Neural Network C extension",
    ext_modules=[lnn_ext],
)
