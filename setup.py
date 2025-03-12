from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "largestinternalrectangle.largestinternalrectangle",  # Package path
        ["largestinternalrectangle/largestinternalrectangle.pyx"],  # File location
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="largestinternalrectangle",
    version="0.1",
    author="Nathan",
    author_email="nathan.tambeur@relu.eu",
    description="Fast Cython implementation for largest interior rectangle calculations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=cythonize(extensions, annotate=True, language_level=3),
    packages=["largestinternalrectangle"],
    include_dirs=[np.get_include()],
    zip_safe=False,
)
