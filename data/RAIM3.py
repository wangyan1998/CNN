import numpy as np
import math

inf = [[17264158.265, 9012094.795, 18535344.120, 23483072.717, -0.000047243624],
       [-4939748.322, 25907897.946, -1938959.799, 21576175.050, -0.000147955131],
       [10411489.279, 15056961.599, 19201346.120, 21308372.478, 0.000388797552],
       [-9251262.428, 11560251.160, 21828403.561, 21671952.866, -0.000049201545]]

inf1 = [[17264158.265, 9012094.795, 18535344.120],
        [-4939748.322, 25907897.946, -1938959.799],
        [10411489.279, 15056961.599, 19201346.120],
        [-9251262.428, 11560251.160, 21828403.561]]

p = [23483072.717, 21576175.050, 21308372.478, 21671952.866]

t = [-0.000047243624, -0.000147955131, 0.000388797552, -0.000049201545]
# 初始估计接收机位置
pos0 = [0, 0, 0]


# pos1 = [-2279829.1069, 5004709.2387, 3219779.0559]


# 获取两点之间的距离
def get_distance1(pos1, pos2):
    v1 = np.array([pos1[0], pos1[1], pos1[2]])
    v2 = np.array([pos2[0], pos2[1], pos2[2]])
    distance = np.linalg.norm(v1 - v2)
    return distance


# 获取修复钟差的伪距
def get_distance2(pos1, pos2, clk):
    v1 = np.array([pos1[0], pos1[1], pos1[2]])
    v2 = np.array([pos2[0], pos2[1], pos2[2]])
    distance = np.linalg.norm(v1 - v2)
    res = distance - (3 * 10 ** 8) * clk
    return res


# 获取估计伪距值
def getp():
    res = []
    i = 0
    while i < len(inf1):
        res.append(get_distance2(pos0, inf1[i], t[i]))
        i = i + 1
    return res


def getdetp1(p):
    res = []
    i = 1
    while i < 4:
        res.append(p[i] - p[0])
        i = i + 1
    return res


# 获取伪距差值
def getdetp(p1):
    res = []
    k = len(p)
    i = 0
    while i < k:
        res.append(p[i] - p1[i])
        i = i + 1
    return res


# 获得每一个r
def getdis():
    res = []
    k = len(p)
    i = 0
    while i < k:
        dis = get_distance1(inf[i], pos0)
        res.append(dis)
        i = i + 1
    return res


# 获取观测矩阵
def getmatH(info, pos, r):
    res = []
    k = len(info)
    i = 0
    c = 3 * 10 ** 8
    while i < k:
        l = []
        l.append((pos[0] - info[i][0]) / r[i])
        l.append((pos[1] - info[i][1]) / r[i])
        l.append((pos[2] - info[i][2]) / r[i])
        i = i + 1
        res.append(l)
    return res


def getH(H):
    res = []
    i = 1
    while i < 4:
        re = []
        re.append(H[i][0] - H[0][0])
        re.append(H[i][1] - H[0][1])
        re.append(H[i][2] - H[0][2])
        res.append(re)
        i = i + 1
    return res


def calTs(p1, H):
    global pos0
    detp1 = getdetp1(p1)
    Ho = getH(H)
    # print(list(Ho))
    H1 = np.array(Ho)
    # print("观测矩阵:\n", H1)
    H2 = np.transpose(H1)
    H3 = np.dot(H2, H1)
    H4 = np.linalg.pinv(H3)
    H5 = np.dot(H1, H4)
    H6 = np.dot(H5, H2)
    In = np.eye(len(p)-1)
    S = In - H6
    R = np.dot(S, detp1)
    Rt = np.transpose(R)
    Ts = np.dot(Rt, R)
    print("Ts:", Ts)


def calresult():
    global pos0
    for j in range(10):
        print(pos0)
        p1 = getp()
        # print("估计位置到各卫星的伪距值为：\n", p1)
        detp = getdetp(p1)
        # print("估计伪距和实际伪距的差：\n", detp)
        r = getdis()
        # print("获取的r为：\n", r)
        H = getmatH(inf1, pos0, r)
        # print("获得的观测矩阵H为:\n", H)
        H1 = np.array(H)
        print("观测矩阵:\n", H1)
        H2 = np.transpose(H1)
        # print(H2)
        H3 = np.dot(H2, H1)
        # print(H3)
        H4 = np.linalg.pinv(H3)
        # print(H4)
        H5 = np.dot(H4, H2)
        # print(H5)
        calTs(p1, H)
        detx = np.dot(H5, detp)
        # print(detx)
        pos0 = pos0 + detx
        # print(pos0)
        # print("位置距离差:\n", pos1 - pos0)


def XYZ_to_LLA(X, Y, Z):
    # WGS84坐标系的参数
    a = 6378137.0  # 椭球长半轴
    b = 6356752.314245  # 椭球短半轴
    ea = np.sqrt((a ** 2 - b ** 2) / a ** 2)
    eb = np.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Z * a, p * b)
    # 计算经纬度及海拔
    longitude = np.arctan2(Y, X)
    latitude = np.arctan2(Z + eb ** 2 * b * np.sin(theta) ** 3, p - ea ** 2 * a * np.cos(theta) ** 3)
    N = a / np.sqrt(1 - ea ** 2 * np.sin(latitude) ** 2)
    altitude = p / np.cos(latitude) - N
    return np.array([np.degrees(latitude), np.degrees(longitude), altitude])


calresult()
solved2 = XYZ_to_LLA(pos0[0], pos0[1], pos0[2])
print(solved2)
