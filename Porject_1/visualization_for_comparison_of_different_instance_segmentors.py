import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader

import train_pointnets_for_semantic_segmentation_radar_scenes
from model.pointnet2_ins_seg import get_gmlp_based_pointnet2_for_instance_segmentation_model, \
    get_pointnet2_for_instance_segmentation_model
from model.pointnet2_sem_seg import get_pointnet2_for_semantic_segmentation_model, \
    get_gmlp_based_pointnet2_for_semantic_segmentation_model
from run_instance_segmentation_for_radar_scenes import pretrained_random_forest_model, \
    remove_duplication_detection_points_with_semantic_information, pretrained_pointnet2_for_semantic_segmentation_model, \
    remove_duplication_detection_points_with_instance_information, pretrained_pointnet2_for_instance_segmentation_model
from utils.radar_scenes_dataset_generator import Radar_Scenes_Test_Dataset

args = train_pointnets_for_semantic_segmentation_radar_scenes.parse_args()
radar_scenes_test_dataset_original_detection_points = Radar_Scenes_Test_Dataset(args.datapath, transforms=None, sample_size=0,
                                                                                LSTM=False, non_static=True)
original_detection_points = DataLoader(radar_scenes_test_dataset_original_detection_points, batch_size=1, shuffle=False, num_workers=0)
original_detection_points = iter(original_detection_points)

RandomForestModelPath = "F:/RadarPointCloudSegmentation/PointNetPorject_V_0/random_forest_classifier_for_clustering_without_semantic_info.m"
classifier = joblib.load(RandomForestModelPath)

radar_scenes_test_dataset_duplicated_detection_points = Radar_Scenes_Test_Dataset(args.datapath, transforms=None, sample_size=200,
                                                                                  LSTM=False, non_static=True)
duplicated_detection_points_dataloader = DataLoader(radar_scenes_test_dataset_duplicated_detection_points,
                                                                                  batch_size=1, shuffle=False, num_workers=0)

# 使用和训练时相同的网络参数
device = torch.device("cpu")

saveloadpath = 'F:/RadarPointCloudSegmentation/PointNetPorject_V_0/pointnet2_semantic_segmentation_validation2_semantic_segmentation_without_gMLP.pth'
detection_points_semantic_segmentor = get_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                              args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                              args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                              args.turn_on_light_weighted_network_model_using_group_conv)
detection_points_semantic_segmentor = detection_points_semantic_segmentor.to(device)
checkpoint = torch.load(saveloadpath, map_location=device)
detection_points_semantic_segmentor.load_state_dict(checkpoint['best_model_state_dict'])

saveloadpath = 'F:/RadarPointCloudSegmentation/PointNetPorject_V_0/pointnet2_semantic_segmentation_validation2_semantic_segmentation_gMLP.pth'
detection_points_semantic_segmentor_gMLP = get_gmlp_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D,
                              args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                              args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, 200, args.tiny_attn,
                              args.turn_on_light_weighted_network_model_using_group_conv)
detection_points_semantic_segmentor_gMLP = detection_points_semantic_segmentor_gMLP.to(device)
checkpoint = torch.load(saveloadpath, map_location=device)
detection_points_semantic_segmentor_gMLP.load_state_dict(checkpoint['best_model_state_dict'])

saveloadpath = 'F:/RadarPointCloudSegmentation/PointNetPorject_V_0/pointnet2_instance_segmentation_validation2_instance_segmentation_without_gMLP.pth'
detection_points_instance_segmentor = get_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                            args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                            args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                            args.turn_on_light_weighted_network_model_using_group_conv)
detection_points_instance_segmentor = detection_points_instance_segmentor.to(device)
checkpoint = torch.load(saveloadpath, map_location=device)
detection_points_instance_segmentor.load_state_dict(checkpoint['best_model_state_dict'])
Th_s = checkpoint['best_model_Ths']

saveloadpath = 'F:/RadarPointCloudSegmentation/PointNetPorject_V_0/pointnet2_instance_segmentation_validation2_instance_segmentation_gMLP.pth'
detection_points_instance_segmentor_gMLP = get_gmlp_based_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D,
                                            args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                            args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, 200,
                                            args.turn_on_light_weighted_network_model_using_group_conv)
detection_points_instance_segmentor_gMLP = detection_points_instance_segmentor_gMLP.to(device)
checkpoint = torch.load(saveloadpath, map_location=device)
detection_points_instance_segmentor_gMLP.load_state_dict(checkpoint['best_model_state_dict'])
Th_s_gMLP = checkpoint['best_model_Ths']

duplicated_detection_points_with_semantic_information_for_all_frames = pretrained_pointnet2_for_semantic_segmentation_model(
    detection_points_semantic_segmentor, duplicated_detection_points_dataloader, args.dataset_D, device)
duplicated_detection_points_with_semantic_information_for_all_frames_gMLP = pretrained_pointnet2_for_semantic_segmentation_model(
    detection_points_semantic_segmentor_gMLP, duplicated_detection_points_dataloader, args.dataset_D, device)
duplicated_detection_points_with_instance_information_for_all_frames = pretrained_pointnet2_for_instance_segmentation_model(
    detection_points_instance_segmentor, duplicated_detection_points_dataloader, args.dataset_D, device, Th_s)
duplicated_detection_points_with_instance_information_for_all_frames_gMLP = pretrained_pointnet2_for_instance_segmentation_model(
    detection_points_instance_segmentor_gMLP, duplicated_detection_points_dataloader, args.dataset_D, device, Th_s_gMLP)

frame_id = 0
marker_list = ['o', 'D', '^', '*', 's']
class_list = ['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE']
assert len(marker_list) == len(class_list)
while True:
    print(frame_id)
    # Run DBSCAN at every frame with only using position information of every detection point, (x, y), as feature.
    detection_points, label = next(original_detection_points)
    pred_instance = DBSCAN(eps=3, min_samples=1).fit_predict(detection_points[0][:, :2])  # Only using position info for DBSCAN.
    pred_classes = pretrained_random_forest_model(classifier, detection_points, pred_instance)

    detection_points = detection_points[0]
    label_id = label[0, 0, :]
    instance_id = label[0, 1, :]
    pred_classes = pred_classes[0]
    fig = plt.figure(figsize=(6, 7))
    fig_a = plt.subplot(321)
    for class_id in range(len(class_list)):
        mask = label_id == class_id
        if not mask.any():
            continue
        points_of_this_class = detection_points[mask]
        x = points_of_this_class[:, 0]
        y = points_of_this_class[:, 1]
        plt.scatter(x, y, c=instance_id[mask], marker=marker_list[class_id], label=class_list[class_id])
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(a)', y=-0.45)
    fig_b = plt.subplot(322)
    N_pred_instances = max(pred_instance) + 1  # 预测实例数
    pred_label = np.zeros_like(pred_instance) - 1  # 记录每个点对应的类别，初始化为全-1
    for instance_id in range(N_pred_instances):
        points_loc_of_this_pred_instance = np.where(pred_instance == instance_id)
        pred_label[points_loc_of_this_pred_instance] = pred_classes[instance_id]  # 该实例的所有点标记为该实例所属类别
    for class_id in range(-1, len(class_list)):  # 未被标记的-1为预测噪声
        mask = pred_label == class_id
        if not mask.any():
            continue
        points_of_this_class = detection_points[mask]
        x = points_of_this_class[:, 0]
        y = points_of_this_class[:, 1]
        if class_id == -1:
            plt.scatter(x, y, c='black', marker='x', label='NOISE')
        else:
            plt.scatter(x, y, c=pred_instance[mask], marker=marker_list[class_id], label=class_list[class_id])
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(b)', y=-0.45)

    # Run DBSCAN at every frame with only using position information of every detection point, (x, y), as feature.
    duplicated_detection_points_with_semantic_information = next(duplicated_detection_points_with_semantic_information_for_all_frames)
    detection_points_with_semantic_information = remove_duplication_detection_points_with_semantic_information(duplicated_detection_points_with_semantic_information)
    detection_points, label, pred_label, pred_center_shift_vectors = detection_points_with_semantic_information
    eps_list = [2.5, 1, 2, 2, 7]
    minpts_list = [1, 1, 1, 1, 1]
    pred_instance = {}  # keys: class_id; values: pred_class
    for class_id in range(args.numclasses):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]  # 属于该类别的点
        features_shift_class = pred_center_shift_vectors[0][mask]  # 属于该类别的点的center shift vectors
        if args.estimate_center_shift_vectors == True:
            pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(
                                                                features_class[:, :2] + features_shift_class[:, :2])
        else:
            pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(features_class[:, :2])
        pred_instance[class_id] = pred_class
    # illustrate predicted labels and instances
    fig_c = plt.subplot(323)
    for class_id in range(len(class_list)):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]
        x = features_class[:, 0]
        y = features_class[:, 1]
        mask = pred_instance[class_id] != -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c=pred_instance[class_id][mask], marker=marker_list[class_id], label=class_list[class_id])
        mask = pred_instance[class_id] == -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c='black', marker='x', label='NOISE')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(c)', y=-0.45)

    # Run DBSCAN at every frame with only using position information of every detection point, (x, y), as feature.
    duplicated_detection_points_with_semantic_information = next(duplicated_detection_points_with_semantic_information_for_all_frames_gMLP)
    detection_points_with_semantic_information = remove_duplication_detection_points_with_semantic_information(duplicated_detection_points_with_semantic_information)
    detection_points, label, pred_label, pred_center_shift_vectors = detection_points_with_semantic_information
    eps_list = [2.5, 1, 2, 2, 7]
    minpts_list = [1, 1, 1, 1, 1]
    pred_instance = {}  # keys: class_id; values: pred_class
    for class_id in range(args.numclasses):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]  # 属于该类别的点
        features_shift_class = pred_center_shift_vectors[0][mask]  # 属于该类别的点的center shift vectors
        if args.estimate_center_shift_vectors == True:
            pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(
                features_class[:, :2] + features_shift_class[:, :2])
        else:
            pred_class = DBSCAN(eps=eps_list[class_id], min_samples=minpts_list[class_id]).fit_predict(features_class[:, :2])
        pred_instance[class_id] = pred_class
    # illustrate predicted labels and instances
    fig_d = plt.subplot(324)
    for class_id in range(len(class_list)):
        mask = pred_label[0, 0, :] == class_id
        if not mask.any():
            continue
        features_class = detection_points[0][mask]
        x = features_class[:, 0]
        y = features_class[:, 1]
        mask = pred_instance[class_id] != -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c=pred_instance[class_id][mask], marker=marker_list[class_id], label=class_list[class_id])
        mask = pred_instance[class_id] == -1
        if mask.any():
            plt.scatter(x[mask], y[mask], c='black', marker='x', label='NOISE')
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(d)', y=-0.45)

    duplicated_detection_points_with_instance_information = next(duplicated_detection_points_with_instance_information_for_all_frames)
    detection_points_with_instance_information = remove_duplication_detection_points_with_instance_information(duplicated_detection_points_with_instance_information)
    detection_points, label, pred_class, pred_instance = detection_points_with_instance_information
    detection_points = detection_points[0]
    label_id = label[0, 0, :]
    instance_id = label[0, 1, :]
    pred_classes = pred_class[0][0]
    pred_instance = pred_instance[0]
    # illustrate predicted labels and instances
    fig_e = plt.subplot(325)
    N_pred_instances = int(max(pred_instance) + 1)  # 预测实例数
    pred_label = np.zeros_like(pred_instance) - 1  # 记录每个点对应的类别，初始化为全-1
    for instance_id in range(N_pred_instances):
        points_loc_of_this_pred_instance = np.where(pred_instance == instance_id)
        pred_label[points_loc_of_this_pred_instance] = pred_classes[instance_id]  # 该实例的所有点标记为该实例所属类别
    for class_id in range(-1, len(class_list)):  # 未被标记的-1为预测噪声
        mask = pred_label == class_id
        if not mask.any():
            continue
        points_of_this_class = detection_points[mask]
        x = points_of_this_class[:, 0]
        y = points_of_this_class[:, 1]
        if class_id == -1:
            plt.scatter(x, y, c='black', marker='x', label='NOISE')
        else:
            plt.scatter(x, y, c=pred_instance[mask], marker=marker_list[class_id], label=class_list[class_id])
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(e)', y=-0.45)

    duplicated_detection_points_with_instance_information = next(duplicated_detection_points_with_instance_information_for_all_frames_gMLP)
    detection_points_with_instance_information = remove_duplication_detection_points_with_instance_information(duplicated_detection_points_with_instance_information)
    detection_points, label, pred_class, pred_instance = detection_points_with_instance_information
    detection_points = detection_points[0]
    label_id = label[0, 0, :]
    instance_id = label[0, 1, :]
    pred_classes = pred_class[0][0]
    pred_instance = pred_instance[0]
    # illustrate predicted labels and instances
    fig_f = plt.subplot(326)
    N_pred_instances = int(max(pred_instance) + 1)  # 预测实例数
    pred_label = np.zeros_like(pred_instance) - 1  # 记录每个点对应的类别，初始化为全-1
    for instance_id in range(N_pred_instances):
        points_loc_of_this_pred_instance = np.where(pred_instance == instance_id)
        pred_label[points_loc_of_this_pred_instance] = pred_classes[instance_id]  # 该实例的所有点标记为该实例所属类别
    for class_id in range(-1, len(class_list)):  # 未被标记的-1为预测噪声
        mask = pred_label == class_id
        if not mask.any():
            continue
        points_of_this_class = detection_points[mask]
        x = points_of_this_class[:, 0]
        y = points_of_this_class[:, 1]
        if class_id == -1:
            plt.scatter(x, y, c='black', marker='x', label='NOISE')
        else:
            plt.scatter(x, y, c=pred_instance[mask], marker=marker_list[class_id], label=class_list[class_id])
    plt.xlabel('x/m')
    plt.ylabel('y/m')
    plt.title('(f)', y=-0.45)

    plt.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.98, wspace=0.30, hspace=0.45)
    plt.show()

    frame_id += 1
