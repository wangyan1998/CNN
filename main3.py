# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


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
out = net(input)
print(out)

# 4.计算损失值
# 一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。有一些不同的损失函数在 nn 包中。
# 一个简单的损失函数就是 nn.MSELoss ，这计算了均方误差。
target = torch.randn(10)
print(target)
target = target.view(1, -1)
print(target)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


