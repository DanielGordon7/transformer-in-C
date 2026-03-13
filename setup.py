from setuptools import setup, Extension

module = Extension(
    "self_attention_module",
    sources=["self_attention_module.c", "self_attention.c"],
    extra_compile_args=["-O2"],
    extra_link_args=["-lm"],
)

setup(
    name="self_attention_module",
    version="1.0",
    description="Scaled dot-product self-attention implemented in C",
    ext_modules=[module],
)
