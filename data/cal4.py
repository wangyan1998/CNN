import math
import torch
import numpy
from scipy.optimize import fsolve

inf = [[-368461.739, 26534822.568, -517664.322, 21966984.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460]]


def get_position(unsolved_value):
    x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
    a = (x - inf[0][0]) ** 2 + (y - inf[0][1]) ** 2 + (z - inf[0][2]) ** 2
    b = (x - inf[1][0]) ** 2 + (y - inf[1][1]) ** 2 + (z - inf[1][2]) ** 2
    c = (x - inf[2][0]) ** 2 + (y - inf[2][1]) ** 2 + (z - inf[2][2]) ** 2
    return [
        math.sqrt(a) - inf[0][3] - (3 * 10 ** 8) * inf[0][4],
        math.sqrt(b) - inf[1][3] - (3 * 10 ** 8) * inf[1][4],
        math.sqrt(c) - inf[2][3] - (3 * 10 ** 8) * inf[2][4]
    ]


def print_revpos1():
    so = fsolve(get_position, [0, 0, 0])
    print("接收机位置", so)
    return so


def get_distance1(pos1):
    v1 = numpy.array([pos1[0], pos1[1], pos1[2]])
    v2 = numpy.array([-2280736.13132096, 5004753.28331651, 3220020.98543618])
    distance = numpy.linalg.norm(v1 - v2)
    return distance


def getgrad(x, y, z):
    k = math.sqrt((x + 2280736.13132096) ** 2 + (y - 5004753.28331651) ** 2 + (z - 3220020.98543618) ** 2)
    n1 = x + 2280736.13132096
    n2 = y - 5004753.28331651
    n3 = z - 3220020.98543618
    print(n1, n2, n3)
    res = []
    res.append(n1 / k)
    res.append(n2 / k)
    res.append(n3 / k)
    # print(res)
    t = torch.tensor(res)
    sign = t.sign()
    return list(numpy.array(sign))


def inter():
    global inf
    dis = 1000000
    while dis > 746:
        r = print_revpos1()
        dis = get_distance1(r)
        print(dis)
        sign = getgrad(r[0], r[1], r[2])
        # sign[0] = -sign[0]
        # sign[1] = -sign[1]
        # sign[2] = -sign[2]
        print(sign)
        i = 0
        while i < 3:
            inf[i][3] = inf[i][3] + 1 * sign[i]
            i = i + 1


inter()
