# cy_largestinternalrectangle

## Installation

To install the package, run:

```bash
pip install cy_largestinternalrectangle
```

## How to Use
```python
import numpy as np
import time
import cy_largestinternalrectangle

# Create a random binary image
img = np.random.randint(-1, 1, (100, 100))

# Measure execution time
t1 = time.time()
print(cy_largestinternalrectangle.largest_interior_rectangle(img > 0))
t2 = time.time()

# Print execution time
print(f"Execution time: {t2 - t1:.6f} seconds")
```




How to Rebuild the Cython Code / Python Package
To compile the Cython code and use it as a Python package, refer to the Cython documentation.

## Compile the Cython Code
Run the following command:

```bash
python setup.py build_ext --inplace
```
## Build the Wheel
To create a distributable wheel package, run:

```bash
python setup.py sdist bdist_wheel
```