import numpy as np

a = np.array([[200.1, 100.2, 330.4, 1.00000],
             [200.2, 100.1, 300.3, 2.0],
             [10.1, 20.2, 30.3, 2.0]], dtype=np.float64)

# print(np.argwhere(a[:,-1] == 2.0)[0,0])
print(a[1, 0:3])

