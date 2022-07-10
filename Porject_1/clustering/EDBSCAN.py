import numpy as np


def EDBSCAN(df, Eps, w1, w2, MinPts):
    """
    EDBSCAN: Elliptical DBSCAN. It ultilizes all features information(spatial info, velocity, RCS) to calculate weighted Euclidean 
    distance for both the outter and inner loop retrieve_neighbors. The searching area is NOT a sphere anymore since the size along
    velocity feature and RCS feature is stretched/compressed by weights. Thus a elliptical searching area is formed rather than sphere.
    See paper: 2015.Modification of DBSCAN and application to rangeDopplerDoA measurements for pedestrian recognition with an automotive 
    radar system, for more info.
    :param:
        df: Set of points, dim=[N,C]
        Eps: Maximum value(Threshold) of all-feature based weighted Euclidean distance
        w1 = weight of velocity distance
        w2 = weight of RCS distance
        MinPts = Minimum number of points within Eps distance of current point, to define current point as core point.
    """
    cluster_label = -1  # 由于sklearn中的DBSCAN算法的cluster编号从0开始，此处是为了统一定义
    NOISE = -1
    UNMARKED = -2
    stack = []
    MinPts -= 1  # 由于sklearn中的DBSCAN算法在统计一个点邻域内点数时包含了该点，而本算法不包含，此处是为了统一定义

    N = df.shape[0]
    # initialize each point with unmarked
    cluster = [UNMARKED] * N

    df = np.column_stack((df, list(range(N))))  # 为每个点添加index

    # for each point in database
    for index in range(N):
        if cluster[index] == UNMARKED:
            neighborhood = retrieve_neighbors(index, df, Eps, w1, w2)
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
                    new_neighborhood = retrieve_neighbors(current_point_index, df, Eps, w1, w2)
                    if len(new_neighborhood) >= MinPts:  # current_point is a new core
                        for neig_index in new_neighborhood:
                            neig_cluster = cluster[int(neig_index)]
                            if neig_cluster == UNMARKED or neig_cluster == NOISE:
                                cluster[int(neig_index)] = cluster_label
                                stack.append(int(neig_index))
    return np.array(cluster)


def retrieve_neighbors(index_center, df, Eps, w1, w2):
    """
        寻找df中第index_center个点的邻域
    """
    center_point = df[index_center, :]
    dist_sq = (df[:, 0] - center_point[0]) * (df[:, 0] - center_point[0]) + \
              (df[:, 1] - center_point[1]) * (df[:, 1] - center_point[1]) + \
              w1 * (df[:, 2] - center_point[2]) * (df[:, 2] - center_point[2]) + \
              w2 * (df[:, 3] - center_point[3]) * (df[:, 3] - center_point[3])
    neighborhood = df[dist_sq <= (Eps * Eps), -1].tolist()
    neighborhood.remove(index_center)
    return neighborhood
