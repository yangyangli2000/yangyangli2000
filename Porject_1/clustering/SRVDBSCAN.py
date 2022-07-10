import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def SRV_DBSCAN(df, Eps1, Eps2, MinPts):
    """
    SRV_DBSCAN: Spatial, RCS, Velocity based DBSCAN. It ultilizes only spatial information to calculate Euclidean distance for 
    the outter loop retrieve_neighbors, as how it is done in standard DBSCAN. Then all features information(spatial, velocity, 
    RCS) are used to form a vector, and calculate vector based Mahananobis distance for inner loop retrieve_neighbors, which 
    is different compare to standard DBSCAN(inner loop retrieve_neighbors in standard DBSCAN is also based on Euclidean distance 
    using spatial information). Such SRV_DBSCAN is inspired by ST-DBSCAN, which could be seen from paper: 2007.ST-DBSCAN An 
    algorithm for clustering spatial temporal data.
    :param:
        df: Set of points, dim=[N,C]
        Eps1: Maximum value(Threshold) of spatial information based Euclidean distance
        Eps2: Maximum value(Threshold) of all-feature(spatial info, velocity, rcs) based Mahananobis distance
        MinPts: Minimum number of points required within Eps1 and Eps2 distance of current point, to define current point as core point.
    """
    cluster_label = -1  # 由于sklearn中的DBSCAN算法的cluster编号从0开始，此处是为了统一定义
    NOISE = -1
    UNMARKED = -2
    stack = []
    MinPts -= 1  # 由于sklearn中的DBSCAN算法在统计一个点邻域内点数时包含了该点，而本算法不包含，此处是为了统一定义

    N = df.shape[0]
    if N == 1:  # 只有1个点，无法计算协方差，单独考虑
        if MinPts <= 0:  # 满足邻域内最少点数条件
            cluster = [0]
        else:
            cluster = [NOISE]
    else:
        # initialize each point with unmarked
        cluster = [UNMARKED] * N

        df = np.column_stack((df, list(range(N))))  # 为每个点添加index

        # for each point in database
        for index in range(N):
            if cluster[index] == UNMARKED:
                neighborhood = retrieve_neighbors(index, df, Eps1, Eps2)
                if len(neighborhood) < MinPts:
                    cluster[index] = NOISE
                else:  # a core point
                    cluster_label = cluster_label + 1
                    cluster[index] = cluster_label  # assign a label to core point
                    for neig_index in neighborhood:  # assign core's label to its neighborhood
                        cluster[int(neig_index)] = cluster_label
                        stack.append(int(neig_index))  # append neighborhood to stack
                    while len(stack) > 0:  # find new neighbors from core point neighborhood
                        current_point_index = stack.pop()
                        new_neighborhood = retrieve_neighbors(current_point_index, df, Eps1, Eps2)
                        if len(new_neighborhood) >= MinPts:  # current_point is a new core
                            for neig_index in new_neighborhood:
                                neig_cluster = cluster[int(neig_index)]
                                if neig_cluster == UNMARKED or neig_cluster == NOISE:
                                    cluster[int(neig_index)] = cluster_label
                                    stack.append(int(neig_index))
    return np.array(cluster)


def retrieve_neighbors(index_center, df, Eps1, Eps2):
    """
        寻找df中第index_center个点的邻域
    """
    center_point = df[index_center, :]

    # filter by time
    # min_time = center_point[2] - Eps2
    # max_time = center_point[2] + Eps2
    # df = df[(df[:, 2] >= min_time) & (df[:, 2] <= max_time), :]

    # filter by all-feature based Mahananobis distance
    dist = mahalanobis_dist(df, index_center)
    df = df[np.array(dist) <= Eps2, :]
    # filter by spatial information based distance
    dist_sq = (df[:, 0]-center_point[0])*(df[:, 0]-center_point[0]) + \
              (df[:, 1]-center_point[1])*(df[:, 1]-center_point[1])
    neighborhood = df[dist_sq <= (Eps1 * Eps1), -1].tolist()
    neighborhood.remove(index_center)
    return neighborhood


def mahalanobis_dist(df, index_center):
    """
        计算点云中所有点到第index_center个点的的马氏距离
    """
    df = df[:, :-1]  # 去掉最后一列（点的序号）
    dfT = df.T  # 求转置[C,N]

    Cov = np.cov(dfT)  # 求协方差矩阵
    if np.linalg.matrix_rank(Cov) <= Cov.shape[0]:  # 协方差矩阵不可逆
        # 对数据做PCA，去掉特征值0的维度
        eig_val, eig_vec = np.linalg.eig(Cov)  # 计算特征值和特征向量
        index = (-eig_val).argsort()  # eig_val从大到小排列
        index = index[eig_val[index] > 1e-3]  # 需要特征值大于0的维度
        P = eig_vec[:, index]  # 降维矩阵P:[C,C']
        df = np.dot(df, P)  # 降维 Y = XP; Y:[N,C']
        Cov = np.cov(df.T)  # 重新计算协方差矩阵

    if Cov.shape == ():  # Cov为1*1矩阵
        invCov = 1 / Cov
    else:
        invCov = np.linalg.inv(Cov)  # 协方差逆矩阵
    center_point = df[index_center]
    dist = [0]*len(df)
    for idx, point in enumerate(df):
        tmp = point - center_point
        dist[idx] = np.sqrt(np.dot(np.dot(tmp, invCov), tmp.T))
    return dist  # dim=N
