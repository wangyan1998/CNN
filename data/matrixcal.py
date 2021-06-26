import numpy as np

H = [[1, 2, 1], [3, 2, 4], [1, 2, 5]]
print(H)
H1 = np.array(H)
print(H1)
H2 = np.transpose(H1)
print(H2)
H4 = np.linalg.pinv(H)  # 矩阵求逆
print(H4)
H5 = H2 * H1
print(H5)
H6 = np.dot(H2, H1)
print(H6)
