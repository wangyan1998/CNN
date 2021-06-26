import torch
import numpy as np
import math

data1 = np.array([
    [21966000, 0, 0, 0, 0, 0, 0, 0],
    [23447000, 0, 0, 0, 0, 0, 0, 0],
    [20154000, 0, 0, 0, 0, 0, 0, 0]])
data2 = np.array([
    [0, 8, 8, 4, 2, 4, 2, 7],
    [0, 0, 2, 2, 1, 1, 3, 6],
    [0, 5, 2, 1, 4, 6, 1, 8]])
data3 = data1 + data2


i = 0
data4 = []
while i < 3:
    data = data3[i][0] + data3[i][1] * 100 + data3[i][2] * 10 + data3[i][3] + data3[i][4] * 0.1 + data3[i][5] * 0.01 + \
           data3[i][6] * 0.001 + data3[i][7] * 0.0001
    data4.append(round(data, 4))
    i = i + 1

print(data4)
