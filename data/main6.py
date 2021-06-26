import torch
import numpy as np
import math

loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

a = np.array([-2279777.96673046, 5004740.32300189, 3219734.86807518])
b = np.array([-2279801.99049722, 5004854.59268167, 3219832.32585892])

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())
loss = math.sqrt(sum(loss))
print(input.float())
print(target.float())

print(torch.tensor([loss]))
