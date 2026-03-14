from setuptools import setup, Extension

module = Extension(
    "transformer_module",
    sources=["transformer_module.c", "transformer.c"],
    extra_compile_args=["-O2", "-std=c99"],
    extra_link_args=["-lm"],
)

setup(
    name="transformer_module",
    version="1.0",
    description=(
        "Pre-LN Transformer (encoder + decoder) implemented in C. "
        "Architecture follows 'Attention Is All You Need' with Pre-Layer "
        "Normalization for improved training stability."
    ),
    ext_modules=[module],
)
