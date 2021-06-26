# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 1.定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 对继承自父类的属性进行初始化

        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


net = Net()
# 2.迭代整个输入
input = torch.randn(1, 1, 32, 32)
target = torch.randn(10)
target = target.view(1, -1)

criterion = nn.MSELoss()
net.zero_grad()  # 清空现存的梯度

# 6.使用优化函数，更新神经网络参数
params = list(net.parameters())
print(len(params))
print(params[0].size())
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()  # zero the gradient buffers

output = net(input)  # 网络处理输入
print(output)
loss = criterion(output, target)  # 计算loss
print(loss)
loss.backward()  # 调用反向传播

optimizer.step()  # Does the update

params = list(net.parameters())
print(len(params))
print(params[0].size())
