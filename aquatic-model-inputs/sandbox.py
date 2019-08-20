import numpy as np
import pandas as pd

q = pd.DataFrame(np.arange(25).reshape(5, 5), columns=['a', 'b', 'c', 'd', 'e'])

a = np.zeros((5, 5))
b = [1, 3, 2, 4, 1]

ncols = 5

out = np.greater.outer(b, np.arange(ncols))
q = q.mask(~out)

print(q)
print(q.sum(axis=1))