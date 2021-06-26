import numpy as np

inf = [[-368461.739, 26534822.568, -517664.322, 21966984.2427, -0.000104647296],
       [10002180.758, 12040222.131, 21796269.831, 23447022.1136, -0.000308443058],
       [-7036480.928, 22592611.906, 11809485.040, 20154521.4618, -0.000038172460],
       [8330122.410, 23062955.196, 10138101.718, 22129309.3677, -0.0002393560],
       [-11000923.048, 24101993.937, -4054413.006, 22292035.2737, -0.000243965]]

inf1 = [[-368461.739, 26534822.568, -517664.322],
        [10002180.758, 12040222.131, 21796269.831],
        [-7036480.928, 22592611.906, 11809485.040],
        [8330122.41, 23062955.196, 10138101.718],
        [-11000923.048, 24101993.937, -4054413.006]]

p = [[21966984.2427], [23447022.1136], [20154521.4618], [22129309.3677], [22292035.2737]]

t = [[-0.000104647296], [-0.000308443058], [-0.000038172460], [-0.000239356], [-0.000243965]]

pos0 = [0, 0, 0]

pos1 = [-2279829.1069, 5004709.2387, 3219779.0559]


def getdetp():
    res = []
    k = len(p)
    i = 1
    while i < k:
        res.append(p[i][0] - p[0][0])
        i = i + 1
    # print(res)
    return res


def get_distance1(pos1, pos2):
    v1 = np.array([pos1[0], pos1[1], pos1[2]])
    v2 = np.array([pos2[0], pos2[1], pos2[2]])
    distance = np.linalg.norm(v1 - v2)
    return distance


# 获得每一个r
def getdis():
    res = []
    k = len(p)
    i = 0
    while i < k:
        dis = get_distance1(inf[i], pos0)
        res.append(dis)
        i = i + 1
    # print(res)
    return res


def getdetdis(dis):
    res = []
    k = len(dis)
    i = 1
    while i < k:
        res.append(dis[i] - dis[0])
        i = i + 1
    # print(res)
    return res


def getmatH(info, pos, r):
    res = []
    k = len(info)
    i = 1
    c = 3 * 10 ** 8
    while i < k:
        l = []
        l.append(((pos[0] - info[i][0]) / r[i]) - ((pos[0] - info[0][0]) / r[0]))
        l.append(((pos[1] - info[i][1]) / r[i]) - ((pos[1] - info[0][1]) / r[0]))
        l.append(((pos[2] - info[i][2]) / r[i]) - ((pos[2] - info[0][2]) / r[0]))
        i = i + 1
        res.append(l)
    return res


def getdata():
    global pos0
    dis1 = 100000
    while dis1 > 1000:
        detp = getdetp()
        dis = getdis()
        H = getmatH(inf1, pos0, dis)
        # print("观测矩阵为H：\n", np.array(H))
        H1 = np.linalg.pinv(H)  # 矩阵求逆
        # print("观测矩阵求逆：\n", H1)
        H2 = np.transpose(H)  # 矩阵转置
        # print("观测矩阵转置：\n", np.transpose(H))
        H3 = np.dot(H2, H)
        # print("中间结果1：\n", H3)
        H4 = np.linalg.pinv(H3)
        # print("中间结果2：\n", H4)
        H5 = np.dot(H4, H2)
        # print("中间结果3：\n", np.array(H5))
        result = np.dot(H5, detp)
        print("pos差值:\n", result)
        dis1 = get_distance1(pos1, result)
        print("和正确位置的差值:\n", dis1)
        pos0[0] += result[0]
        pos0[1] += result[1]
        pos0[2] += result[2]
        print("pos0为：", pos0)
    print(dis1)
    return dis1


getdata()
