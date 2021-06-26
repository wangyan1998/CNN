import math

import torch
import numpy as np


def f(x, y, z):
    return math.sqrt((x + 2280736.13132096) ** 2 + (y - 5004753.28331651) ** 2 + (z - 3220020.98543618) ** 2)


def f_prime(x, y, z, f):
    return torch.autograd.grad(f(x, y, z), x, create_graph=True)


x1 = torch.tensor([10.], requires_grad=True)
y1 = torch.tensor([10.], requires_grad=True)
z1 = torch.tensor([10.], requires_grad=True)

for i in range(100):
    f_val = f(x1, y1, z1)
    fp = f_prime(x1, y1, z1, f)[0]
    x1 = x1 - f_val / fp

print("solution:", x1)
print("final function value:", f(x1, y1, z1))
