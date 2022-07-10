'''
PointNet++ for Instance Segmentation.

NOMENCLATURE:
    B is the number of data in each batch
    N is the total number of input points
    C is the dimension of feature or channels
    D is the dimension of coordinate,

    npoint = the number of centers
    nsample = the number of points sampled from each circle 

The dimension of C and D can be extremely confusing, In our code, the D, dimension of the coordinate is 2, x_cc and y_cc and C, the dimension of feature is 2, vr_compensated and rcs.

For more details of these two functions, please refer to Figure 2 of the original paper.
For a detailed explanation concerning parameter choice, please refer to README.md.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, gmlp_block


class get_pointnet2_for_instance_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample,
                 second_layer_npoint, second_layer_radius, second_layer_nsample, group_conv=False):
        super(get_pointnet2_for_instance_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])  # l0_points维度为2

            self.conv_sim = nn.Conv1d(16, 16, 1, groups=4)

            self.conv_cf = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv1 = nn.Conv1d(16, 1, 1)

            self.conv_sem = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(16, num_class, 1)
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])  # l0_points维度为2

            self.conv_sim = nn.Conv1d(16, 16, 1)

            self.conv_cf = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv1 = nn.Conv1d(16, 1, 1)

            self.conv_sem = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(16, num_class, 1)

    def forward(self, xyz, dataset_D):
        l0_points = xyz[:, dataset_D:, :]  # [B, C, N]
        l0_xyz = xyz[:, :dataset_D, :]  # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # go through the first set abstraction layer, where the feature dimension goes from C to 64
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # go through the second set abstraction layer, where the feature dimension goes from 64 to 256

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # go through the first feature propagation layer, skip link l2 with l1
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # go through the second feature propagation layer, skip link l1 with l0

        # l0_points is the feature map of points which is learned by PointNet++, which is denoted by F in paper "2018.SGPN: 
        # Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation", and it will diverge into three branches 
        # that each pass F through a single PointNet layer to obtain sized N × C feature matrices F_sim, F_cf , F_sem, whose 
        # dimension is "number of points" by "number of features per point", and which will be used to compute a similarity 
        # matrix, a confidence map and a semantic segmentation map respectively.
        # Refer to Section 3.1 of paper "2018.SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation" 
        # for more information.
        # 代码参考：https://github.com/laughtervv/SGPN/blob/master/models/model.py
        
        # Calculate similarity matrix (S). Each element in the similarity matrix is actual the L2 norm(It is indeed a Euclidean 
        # distance) of difference between the feature vector belongs to point i and the feature vector belongs to point j. Thus 
        # the smaller the element in the similarity matrix the more likelihood point i and j belong to the same instance.
        # TODO: Should we try with other metrics here rather than Euclidean distance here? e.g. Mahananobis distance, inner product, 
        # cosine similarity, etc. (Beware what we have here is "feautre vector of every point", the "features" here do NOT have
        # any explainable physical meaning, thus we do not need to always use "distance" with physical meaning. Any metric could
        # be applied to measure the "similarity" between two vectors, e.g. inner product, cosine similarity of two vectors might
        # be suitable.)
        F_sim = self.conv_sim(l0_points)
        F_sim = F_sim.permute(0, 2, 1)  # [B, N, C]
        batch_size = F_sim.shape[0]
        # Beware here the calculation of similarity matrix(Euclidean distance of difference between the feature vector belongs 
        # to point i and the feature vector belongs to point j) is represented by A, B, and C. See more detail here：https://blog.csdn.net/qq_26667429/article/details/101557821
        A = (F_sim * F_sim).sum(dim=2).reshape((batch_size, -1, 1))  # [B, N, 1]
        C = A.permute(0, 2, 1)  # [B, 1, N]
        B = 2 * F_sim.matmul(F_sim.permute(0, 2, 1))  # [B, N, N]
        sim_mat = A - B + C
        sim_mat = torch.clamp(sim_mat, 0)  # 由于浮点数计算误差，求出来的similarity值可能小于0（例如 -1e-8），我们将这些数都设置为0以减小训练误差

        # Calculate similarity confidence map(The map of confidence score for similarity).
        x = self.drop1(F.relu(self.bn1(self.conv_cf(l0_points))))
        x = self.conv1(x).squeeze(1)  # [B, 1, N]->[B,N]
        cf_mat_for_similarity = x.sigmoid()  # limit the value range of each element to [0,1]

        # Calculate semantic segmentation map.
        x = self.drop2(F.relu(self.bn2(self.conv_sem(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        sem_mat = x.permute(0, 2, 1)  # [B, N, num_class]

        pred = (sim_mat, cf_mat_for_similarity, sem_mat)
        return pred

class get_gmlp_based_pointnet2_for_instance_segmentation_model(nn.Module):
    def __init__(self, num_class, dataset_D, dataset_C, first_layer_npoint, first_layer_radius, first_layer_nsample,
                 second_layer_npoint, second_layer_radius, second_layer_nsample, number_of_points_in_a_frame, group_conv=False):
        super(get_gmlp_based_pointnet2_for_instance_segmentation_model, self).__init__()
        if group_conv == True:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 2, 8], False)
            self.gmlp1 = gmlp_block(64, 128, first_layer_npoint)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [2, 16, 32], False)
            self.gmlp2 = gmlp_block(256, 512, second_layer_npoint)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [16, 8])
            self.gmlp3 = gmlp_block(32, 64, first_layer_npoint)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [2, 8, 4])  # l0_points维度为2
            self.gmlp4 = gmlp_block(16, 32, number_of_points_in_a_frame)

            self.conv_sim = nn.Conv1d(16, 16, 1, groups=4)

            self.conv_cf = nn.Conv1d(16, 16, 1, groups=4)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv1 = nn.Conv1d(16, 1, 1)

            self.conv_sem = nn.Conv1d(16, 16, 1, groups=4)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(16, num_class, 1)
        else:
            self.sa1 = PointNetSetAbstraction(first_layer_npoint, first_layer_radius, first_layer_nsample,
                                              dataset_C + dataset_D, [first_layer_nsample, 32, 64], [1, 1, 1], False)
            self.gmlp1 = gmlp_block(64, 128, first_layer_npoint)
            self.sa2 = PointNetSetAbstraction(second_layer_npoint, second_layer_radius, second_layer_nsample,
                                              64 + dataset_D, [64, 128, 256], [1, 1, 1], False)
            self.gmlp2 = gmlp_block(256, 512, second_layer_npoint)
            self.fp2 = PointNetFeaturePropagation(256 + 64, [64, 32], [1, 1])
            self.gmlp3 = gmlp_block(32, 64, first_layer_npoint)
            self.fp1 = PointNetFeaturePropagation(32 + 2, [32, 32, 16], [1, 1, 1])  # l0_points维度为2
            self.gmlp4 = gmlp_block(16, 32, number_of_points_in_a_frame)

            self.conv_sim = nn.Conv1d(16, 16, 1)

            self.conv_cf = nn.Conv1d(16, 16, 1)
            self.bn1 = nn.BatchNorm1d(16)
            self.drop1 = nn.Dropout(0.5)
            self.conv1 = nn.Conv1d(16, 1, 1)

            self.conv_sem = nn.Conv1d(16, 16, 1)
            self.bn2 = nn.BatchNorm1d(16)
            self.drop2 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(16, num_class, 1)

    def forward(self, xyz, dataset_D):
        l0_points = xyz[:, dataset_D:, :]  # [B, C, N]
        l0_xyz = xyz[:, :dataset_D, :]  # [B, D, N], the splicing terminal index is not inclusive
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # go through the first set abstraction layer, where the feature dimension goes from C to 64
        l1_points = self.gmlp1(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # go through the second set abstraction layer, where the feature dimension goes from 64 to 256
        l2_points = self.gmlp2(l2_points.permute(0, 2, 1)).permute(0, 2, 1)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # go through the first feature propagation layer, skip link l2 with l1
        l1_points = self.gmlp3(l1_points.permute(0, 2, 1)).permute(0, 2, 1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # go through the second feature propagation layer, skip link l1 with l0
        l0_points = self.gmlp4(l0_points.permute(0, 2, 1)).permute(0, 2, 1)

        # l0_points is the feature map of points which is learned by PointNet++, which is denoted by F in paper "2018.SGPN:
        # Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation", and it will diverge into three branches
        # that each pass F through a single PointNet layer to obtain sized N × C feature matrices F_sim, F_cf , F_sem, whose
        # dimension is "number of points" by "number of features per point", and which will be used to compute a similarity
        # matrix, a confidence map and a semantic segmentation map respectively.
        # Refer to Section 3.1 of paper "2018.SGPN: Similarity Group Proposal Network for 3D Point Cloud Instance Segmentation"
        # for more information.
        # 代码参考：https://github.com/laughtervv/SGPN/blob/master/models/model.py

        # Calculate similarity matrix (S). Each element in the similarity matrix is actual the L2 norm(It is indeed a Euclidean
        # distance) of difference between the feature vector belongs to point i and the feature vector belongs to point j. Thus
        # the smaller the element in the similarity matrix the more likelihood point i and j belong to the same instance.
        # TODO: Should we try with other metrics here rather than Euclidean distance here? e.g. Mahananobis distance, inner product,
        # cosine similarity, etc. (Beware what we have here is "feautre vector of every point", the "features" here do NOT have
        # any explainable physical meaning, thus we do not need to always use "distance" with physical meaning. Any metric could
        # be applied to measure the "similarity" between two vectors, e.g. inner product, cosine similarity of two vectors might
        # be suitable.)
        F_sim = self.conv_sim(l0_points)
        F_sim = F_sim.permute(0, 2, 1)  # [B, N, C]
        batch_size = F_sim.shape[0]
        # Beware here the calculation of similarity matrix(Euclidean distance of difference between the feature vector belongs
        # to point i and the feature vector belongs to point j) is represented by A, B, and C. See more detail here：https://blog.csdn.net/qq_26667429/article/details/101557821
        A = (F_sim * F_sim).sum(dim=2).reshape((batch_size, -1, 1))  # [B, N, 1]
        C = A.permute(0, 2, 1)  # [B, 1, N]
        B = 2 * F_sim.matmul(F_sim.permute(0, 2, 1))  # [B, N, N]
        sim_mat = A - B + C
        sim_mat = torch.clamp(sim_mat, 0)  # 由于浮点数计算误差，求出来的similarity值可能小于0（例如 -1e-8），我们将这些数都设置为0以减小训练误差

        # Calculate similarity confidence map(The map of confidence score for similarity).
        x = self.drop1(F.relu(self.bn1(self.conv_cf(l0_points))))
        x = self.conv1(x).squeeze(1)  # [B, 1, N]->[B,N]
        cf_mat_for_similarity = x.sigmoid()  # limit the value range of each element to [0,1]

        # Calculate semantic segmentation map.
        x = self.drop2(F.relu(self.bn2(self.conv_sem(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        sem_mat = x.permute(0, 2, 1)  # [B, N, num_class]

        pred = (sim_mat, cf_mat_for_similarity, sem_mat)
        return pred

class get_loss(nn.Module):
    # Calculate the loss for PointNet++ based instance segmentation using radar detection points.
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, label, device, alpha=10., margin=[1., 2.]):  # margin代表SGPN论文中double hinge loss的K1,K2
        class_gt = label[:, 0, :]  # [B, N]
        instance_gt = label[:, 1, :]  # [B, N]
        pred_simmat, pred_cfmat, pred_semmat = pred
        B = label.shape[0]
        N = label.shape[2]

        # 代码参考：https://github.com/laughtervv/SGPN/blob/master/models/model.py
        # groundtruth group (G) G_ij表示第i,j个点是否属于同一实例
        gt_group = torch.zeros((B, N, N)).to(device)
        # groundtruth semantic (Gs) Gs_ij表示第i,j个点是否属于同一类别
        gt_sem = torch.zeros((B, N, N)).to(device)
        for batch_id in range(B):
            N_instance = int(max(instance_gt[batch_id]) + 1)
            for instance_id in range(N_instance):
                point_loc_of_this_instance = torch.where(instance_gt[batch_id] == instance_id)[0]  # 属于该实例的点的位序
                gt_group_of_this_instance = torch.zeros(N).to(device)
                gt_group_of_this_instance[point_loc_of_this_instance] = 1  # 位序所对应元素为1（如第1,2,4个点属于该实例，则第1,2,4个元素为1，其余为0）
                gt_group[batch_id] += gt_group_of_this_instance.view(-1, 1) * gt_group_of_this_instance  # 设置G中对应行对应列为1
            classes = set([int(class_id) for class_id in class_gt[batch_id]])
            for class_id in classes:
                point_loc_of_this_class = torch.where(class_gt[batch_id] == class_id)[0]  # 属于该实例的点的位序
                gt_sem_of_this_class = torch.zeros(N).to(device)
                gt_sem_of_this_class[point_loc_of_this_class] = 1  # 位序所对应元素为1
                gt_sem[batch_id] += gt_sem_of_this_class.view(-1, 1) * gt_sem_of_this_class  # 设置Gs中对应行对应列为1
        samegroup_mat_label = gt_group  # 第i行第j列元素表示第i,j个点是否属于同一实例
        diffgroup_samesem_mat_label = (1 - gt_group).mul(gt_sem)  # 第i行第j列元素表示第i,j个点是否不属于同一实例但属于同一类
        diffgroup_diffsem_mat_label = (1 - gt_group).mul(1 - gt_sem)  # 第i行第j列元素表示第i,j个点是否不属于同一类

        # Loss part 1: Calculate similarity loss by using double hinge loss.
        sim_class_1_loss = pred_simmat.mul(samegroup_mat_label)  # [B,N,N]
        sim_class_2_loss = alpha * diffgroup_samesem_mat_label.mul(torch.clamp(margin[0] - pred_simmat, 0))
        sim_class_3_loss = diffgroup_diffsem_mat_label.mul(torch.clamp(margin[1] - pred_simmat, 0))
        simmat_loss = sim_class_1_loss + sim_class_2_loss + sim_class_3_loss  # [B,N,N]
        sim_loss = simmat_loss.mean()  # 平均每个batch、每对点的simmat_loss

        # Loss part 2: Calculate similarity confidence map(the confidence of predicted similarity distance vector for each detection point) loss.
        # See subsection "Similarity Confidence Map" in section 3.1. of paper "2018.SGPN: Similarity Group Proposal Network for 3D 
        # Point Cloud Instance Segmentation" for more detail.
        # TODO: Such kind of way(For each row in Si, we expect the ground-truth value in the similarity confidence map CM_i to be the 
        # intersection over union (IoU) between the set of points in the predicted group Si and the ground truth group Gi. Our loss
        # cf_loss_for_similarity is the L2 loss between the inferred and expected CM) to define the similarity confidence loss may NOT be the suitable
        # one, is there any better way to describe the confidence score(uncertainty) modeling of this "clustering" problem? Beware here
        # it seems like a problem of modeling of uncertainty for classification task(as it is estimation of ID per point), but indeed 
        # it is NOT the actual modeling of uncertainty for classification task since the total number of IDs is NOT known in advance.
        # Such information is only available as long as the predicted similarity matrix is translated into predicted instances.
        gt_group = torch.gt(gt_group, 0.5)  # 1->true, 0->false, [B,N,N]
        pred_group = torch.lt(pred_simmat, margin[0])  # 小于margin[0]->true, 大于margin[0]->false, [B,N,N]
        union = torch.logical_or(gt_group, pred_group).float().sum(axis=2)  # [B, N]
        intersection = torch.logical_and(gt_group, pred_group).float().sum(axis=2)  # [B, N]
        cf_gt = intersection / union  # 求IoU值作为conf_map的groundtruth值，dim=[B, N]
        cf_loss_for_similarity = (cf_gt - pred_cfmat).norm(dim=1)  # [B]
        cf_loss_for_similarity = cf_loss_for_similarity.sum() / B
        # Beware there is an alternative way of calculating cf_loss_for_similarity by using binary cross entropy loss showing as below
        # (如采用该计算方式，将此loss项的权重设置为2).
        """ cf_loss_for_similarity = F.binary_cross_entropy(pred_cfmat, cf_gt) """

        # Loss part 3: Calculate semantic segmentation loss by using cross entropy loss(Beware since the predicted semantic matrix, pred_semmat, has already 
        # been the intermedian output from softmax, thus here we only use nll_loss rather than the full cross_entropy_loss which contains the operation of
        # softmax and nll).
        sem_loss = 0
        for batch_id in range(B):
            sem_loss += F.nll_loss(pred_semmat[batch_id], class_gt[batch_id].long().to(device))
        sem_loss = sem_loss / B

        '''# Loss part 4: Calculate IoU loss.
        # if a row in similarity matrix (after mapping each similarity distance to [0,1]) is [0.9, 0, 0.6, 0.1, 0.2]
        # (then the score is [0.1, 1, 0.4, 0.9, 0.8]), and the corresponding groundtruth is [0, 1, 1, 0, 1],
        # assume that the threshold is 0.5, then the second and the fifth points belong to both the predicted instance
        # and the groundtruth instance, the third point only belongs to the groundtruth instance, and the forth point only
        # belongs to the predicted instance. So the IoU between predicted instance and groundtruth instance can be calculated
        # as follows:
        # intersection = (1+1)/2 + (0.8+1)/2
        # union = (1+1)/2 + 1 + 0.9 + (0.8+1)/2
        # iou = intersection / union
        intersection_mat = torch.zeros_like(pred_simmat).to(device)
        union_mat = torch.zeros_like(pred_simmat).to(device)
        pred_simmat = 1 - pred_simmat.atan() * 2 / math.pi  # 将0到无穷的数映射到[0,1]区间；由于simmat中0代表相同，无穷大为完全不同，所以映射后需取反（1代表相同，0代表不同）
        intersection_loc = torch.where(gt_group & pred_group)  # 属于交集的点的位序
        point_belonging_to_pred_instance_only_loc = torch.where(pred_group & ~gt_group)  # 属于预测实例而不属于实际实例的点的位序
        point_belonging_to_gt_instance_only_loc = torch.where(~pred_group & gt_group)  # 属于实际实例而不属于预测实例的点的位序
        intersection_mat[intersection_loc] = (pred_simmat[intersection_loc] + 1) / 2  # 交集中每个点的分数 = (预测分数 + 1) / 2，其中1代表实际值
        union_mat += intersection_mat
        union_mat[point_belonging_to_pred_instance_only_loc] = pred_simmat[point_belonging_to_pred_instance_only_loc]
        union_mat[point_belonging_to_gt_instance_only_loc] = 1  # 1代表实际值
        iou = intersection_mat.sum(dim=2) / union_mat.sum(dim=2)  # [B,N]
        iou_loss = 1 - iou.mean()  # iou值越接近1越好'''

        return sim_loss, cf_loss_for_similarity, sem_loss
