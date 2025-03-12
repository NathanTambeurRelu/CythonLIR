import numpy as np
import time
import cy_largestinternalrectangle


img = np.random.randint(-1, 1, (100, 100))
t1 = time.time()
print((cy_largestinternalrectangle.largest_interior_rectangle(img>0)))
t2 = time.time()
print(t2-t1)
