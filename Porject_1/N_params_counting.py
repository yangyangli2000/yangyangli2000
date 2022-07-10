from thop import profile, clever_format
from torchsummary import summary
import time

import joblib
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import train_pointnets_for_semantic_segmentation_radar_scenes
from model.pointnet2_ins_seg import get_pointnet2_for_instance_segmentation_model, get_gmlp_based_pointnet2_for_instance_segmentation_model
from model.pointnet2_sem_seg import get_pointnet2_for_semantic_segmentation_model, \
    get_gmlp_based_pointnet2_for_semantic_segmentation_model, \
    get_external_attention_based_pointnet2_for_semantic_segmentation_model, \
    get_self_attention_based_pointnet2_for_semantic_segmentation_model
from model.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from run_instance_segmentation_for_radar_scenes import pretrained_random_forest_model, \
    remove_duplication_detection_points_with_semantic_information, pretrained_pointnet2_for_semantic_segmentation_model, \
    remove_duplication_detection_points_with_instance_information, pretrained_pointnet2_for_instance_segmentation_model
from utils.radar_scenes_dataset_generator import Radar_Scenes_Test_Dataset

args = train_pointnets_for_semantic_segmentation_radar_scenes.parse_args()

# 使用和训练时相同的网络参数
device = torch.device("cpu")

# 不带gMLP语义分割
detection_points_semantic_segmentor = get_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                                                                    args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                    args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                                                                    args.turn_on_light_weighted_network_model_using_group_conv)
torch.save(detection_points_semantic_segmentor.state_dict(), 'model_sem.pth')
summary(detection_points_semantic_segmentor, (4, 200))

# 带gMLP语义分割
detection_points_semantic_segmentor_gMLP = get_gmlp_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D,
                                                                                                    args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                                    args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, 200, args.tiny_attn,
                                                                                                    args.turn_on_light_weighted_network_model_using_group_conv)
torch.save(detection_points_semantic_segmentor_gMLP.state_dict(), 'model_sem_gMLP.pth')
summary(detection_points_semantic_segmentor_gMLP, (4, 200))

# 不带gMLP实例分割
detection_points_instance_segmentor = get_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                                                                    args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                    args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                                                                    args.turn_on_light_weighted_network_model_using_group_conv)
torch.save(detection_points_instance_segmentor.state_dict(), 'model_ins.pth')
summary(detection_points_instance_segmentor, (4, 200))

# 带gMLP实例分割
detection_points_instance_segmentor_gMLP = get_gmlp_based_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D,
                                                                                                    args.dataset_C, args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                                    args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, 200)
torch.save(detection_points_instance_segmentor_gMLP.state_dict(), 'model_ins_gMLP.pth')
summary(detection_points_instance_segmentor_gMLP, (4, 200))

# 带external attention语义分割
detection_points_semantic_segmentor_ea = get_external_attention_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                                                                                             args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                                             args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                                                                                             args.turn_on_light_weighted_network_model_using_group_conv)
torch.save(detection_points_semantic_segmentor_ea.state_dict(), 'model_sem_ea.pth')
summary(detection_points_semantic_segmentor_ea, (4, 200))

# 带self attention语义分割
detection_points_semantic_segmentor_sa = get_self_attention_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                                                                                                args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                                                args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                                                                                                args.turn_on_light_weighted_network_model_using_group_conv)
torch.save(detection_points_semantic_segmentor_sa.state_dict(), 'model_sem_sa.pth')
summary(detection_points_semantic_segmentor_sa, (4, 200))

# 带aMLP语义分割
detection_points_semantic_segmentor_aMLP = get_gmlp_based_pointnet2_for_semantic_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                                                                                            args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                                                                                            args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                                                                                            200, tiny_attn=True)
torch.save(detection_points_semantic_segmentor_aMLP.state_dict(), 'model_sem_aMLP.pth')
summary(detection_points_semantic_segmentor_aMLP, (4, 200))