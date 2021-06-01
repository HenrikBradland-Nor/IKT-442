import numpy as np


a = np.random.randint(0,8,[3,3,3])


b = np.array([1, 1])
print(a[tuple(b)])

for val in b:
    a = a[val]

print(a)