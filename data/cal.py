import numpy as np

mat = np.array(
    [[6, 3, 5],
     [5, 2, 8],
     [4, 1, 7]])
mat_inv = np.linalg.pinv(mat)  # 矩阵求逆
print(mat_inv)
mat_trp = np.transpose(mat)    # 矩阵转置
print(mat_trp)
offset = np.array([[16, 128, 128]])
print(np.transpose(offset))
print(np.eye(5))

