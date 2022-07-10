import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.radar_scenes_dataset_generator import Radar_Scenes_Train_Dataset, Radar_Scenes_Validation_Dataset


def ExtractFeatures(detection_points):
    x = detection_points[:, 0]
    y = detection_points[:, 1]
    v = detection_points[:, 2]
    RCS = detection_points[:, 3]
    range = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    RCS_mean = RCS.mean()
    RCS_std = np.array(RCS).std()
    RCS_spread = max(RCS) - min(RCS)
    v_max = max(v)
    v_min = min(v)
    range_min = min(range)
    range_max = max(range)
    range_mean = range.mean()
    range_std = np.array(range).std()
    ang_spread = max(angle) - min(angle)
    ang_std = np.array(angle).std()
    hist_v = np.histogram(v, bins=10)[0]
    hist_RCS = np.histogram(RCS, bins=10)[0]
    # The two eigenvalues of the covariance matrix of x and y, proposed in paper: "2018. Comparison of Random Forest and Long
    # Short-Term Memory Network Performances in Classification Tasks Using Radar". The eigenvalues represent the variance of
    # the data along the eigenvector directions, so here we expect the random forest classifier should "learn" something related
    # to "shape information of single object"
    eig_cov_xy = np.sort(np.linalg.eig(np.cov(np.vstack((x, y)), bias=True))[0])
    return np.hstack((RCS_mean, RCS_std, RCS_spread, v_max, v_min, range_min, range_max, range_mean, range_std, ang_spread,
                      ang_std, hist_v, hist_RCS, eig_cov_xy))


if __name__ == '__main__':
    """ datapath = 'D:/RadarScenes/radar_scenes_processed_data' """
    datapath = 'D:/Tech_Resource/Paper_Resource/Dataset/RadarScenes/RadarScenes/data'
    radar_scenes_train_dataset = Radar_Scenes_Train_Dataset(datapath, transforms=None, sample_size=0, LSTM=False, non_static=True)
    train_dataloader = DataLoader(radar_scenes_train_dataset, batch_size=1, shuffle=False, num_workers=0)

    train_features = []
    train_targets = []
    print("Start to train random forest classifier by using training data.")
    for detection_points, label in tqdm(train_dataloader):  # detection_points:[B,N,C] C:x,y,v,rcs
        class_gt = label[0, 0, :]
        instance_gt = label[0, 1, :]
        N_instances = int(max(instance_gt) + 1)
        for instance_id in range(N_instances):
            points_loc_of_this_instance = np.where(instance_gt == instance_id)[0]  # 所有属于该实例的点的位序
            points_of_this_instance = detection_points[0][points_loc_of_this_instance]  # 所有属于该实例的点
            features_of_this_instance = ExtractFeatures(points_of_this_instance)
            class_of_this_instance = class_gt[points_loc_of_this_instance[0]]  # 该实例所属类别
            train_features.append(features_of_this_instance)
            train_targets.append(class_of_this_instance)
    classifier = RandomForestClassifier()
    classifier.fit(train_features, train_targets)
    """ joblib.dump(classifier, "random_forest_classifier_for_clustering_without_semantic_info.m") """
    joblib.dump(classifier, "D:/Tech_Resource/Paper_Resource/Perception_R_or_RC_Fusion_with_BingZhu_Project/Projects/Project_1/pre_trained_model/random_forest_classifier_for_clustering_without_semantic_info.m")

    # classifier = joblib.load("random_forest_classifier_for_clustering_without_semantic_info.m")
    # pred_classes = classifier.predict(pred_instances)

    '''Validation'''
    radar_scenes_validation_dataset = Radar_Scenes_Validation_Dataset(datapath, transforms=None, sample_size=0, LSTM=False, non_static=True)
    validation_dataloader = DataLoader(radar_scenes_validation_dataset, batch_size=1, shuffle=False, num_workers=0)

    validation_features = []
    validation_targets = []
    print("Start to validate random forest classifier by using validation data.")
    for detection_points, label in tqdm(validation_dataloader):  # detection_points:[B,N,C] C:x,y,v,rcs
        class_gt = label[0, 0, :]
        instance_gt = label[0, 1, :]
        N_instances = int(max(instance_gt) + 1)
        for instance_id in range(N_instances):
            points_loc_of_this_instance = np.where(instance_gt == instance_id)[0]  # 所有属于该实例的点的位序
            points_of_this_instance = detection_points[0][points_loc_of_this_instance]  # 所有属于该实例的点
            features_of_this_instance = ExtractFeatures(points_of_this_instance)
            class_of_this_instance = class_gt[points_loc_of_this_instance[0]]  # 该实例所属类别
            validation_features.append(features_of_this_instance)
            validation_targets.append(class_of_this_instance)
    acc = classifier.score(validation_features, validation_targets)
    print("The accuracy for validation data is ", acc)
