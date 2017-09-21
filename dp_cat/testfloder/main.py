import numpy as np

x = np.array([

[
[1, 2,3], [3,3,3], [3,3,4]
], 

[
[1,3, 2], [3,3,3], [3,3,3]
]

])
print(x)
print(x.shape)

y= np.array([[1,2,3,4]])
print(y)
print(y.shape)
# 矩阵中 ':' 代表所有行/列
print(x[:,0])
