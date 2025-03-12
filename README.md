### How to rebuild the Cython code/Python package

For compilation of Cython code and usage as Python Package see:
https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html

run: python setup.py build_ext --inplace

Building of the wheel:
python setup.py sdist bdist_wheel