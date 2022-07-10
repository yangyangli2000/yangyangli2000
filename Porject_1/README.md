# TODO
1. ~~找到出问题的地方~~ 已完成
2. ~~查阅资料：为什么小的值通过Convolution后会生成0.~~
3. ~~查阅资料：为什么0通过Batch Norm后会生成NaN~~
4. ~~加入Cosine Decay作为选项~~ 已完成
5. ~~找出更好的问题处理方法~~  已完成
6. ~~在北航高算上跑起来~~ 已完成
7. batchsize和learning rate,以及learning rate的减小需要再优化。在高算上显存几乎处于不受限的状态，batsize测试了1024，完全没有问题。但当batch_size变化的时候，learning rate的策略也需要变化，需要实验什么是最好的方案
8. ~~validation现在是每10个epoch才做一次，但comment写的是每次都做，这块是我的疏忽。validation这块需要重新调整,指标调整为mIoU。~~  已完成
9. ~~为什么linux系统和win系统跑起来会不同， 这个很怪~~

# 北航高算
1. 调用工程师配置的环境testenv
2. srun -p gpu-normal --gres=gpu:1 python train_pointnets_radar_scenes.py 用normal其实就可以了，如果人很多我们可以用high，速度没有差别，唯一的差别是排队的时候可以插队,价格normal 3.75 high 5

# Table of Content
- [Project Overview](#project-overview)
- [Code Overview](#code-overview)
  * [Data Processing](#data-processing)
  * [PointNet](#pointnet)
  * [PointNet++](#pointnet--)
- [Dataset Overview](#dataset-overview)
  * [Website:](#website-)
  * [Dataset Composition:](#dataset-composition-)
  * [Data Vitualization](#data-vitualization)
  * [Radar Sensor Position](#radar-sensor-position)
  * [Composition of Data](#composition-of-data)
    + [Odometry](#odometry)
    + [Radar](#radar)
  * [Snippet of Radar Data meta-information](#snippet-of-radar-data-meta-information)
  * [Relationship between Timestamps and Scene](#relationship-between-timestamps-and-scene)
  * [Labels](#labels)

## Project Overview
There are two objectives of this project:
1. feed each point to PointNet++ and puke out semantic information. Then with that semantic inforamtion, we would run various machine learning algs for clustering. Where the true value is label_id.
2. feed each point to PointNet++ and puke out instance segmentation, where the true values are both label_id & track_id.

## Code Overview

### Data Processing
![1](https://github.com/BaiLiping/Radar_PointNet_InstanceSegmentation/blob/main/PointNetPorject_V_0/radar_scenes_description/Project1_Overview.png)

### PointNet
The code for PointNet is adapted from the latest implementation of PointNet with Pytorch https://github.com/yanx27/Pointnet_Pointnet2_pytorc


![1](https://github.com/BaiLiping/Radar_PointNet_InstanceSegmentation/blob/main/PointNetPorject_V_0/radar_scenes_description/PointNet.png)

### PointNet++
The code is PointNet++ adapted from the latest implementation of PointNet++ with Pytorch https://github.com/yanx27/Pointnet_Pointnet2_pytorc. The original notation concerning C and D in that repository are erroneous, so be advised to pay particular attention when it comes to sorting out the dimension of your matrix.

**From PointNet to PointNet++:**
The basic idea of PointNet is to learn a spatial encoding of each point and then aggregate all individual point features to a global point cloud signature. By its design, PointNet doesnot capture local structure induced by the metric.

However, exploiting local structure has proven tobe important for the success of convolutional architectures. A CNN takes data defined on regulargrids as the input and is able to progressively capture features at increasingly larger scales along a multi-resolution hierarchy. At lower levels neurons have smaller receptive fields whereas at higherlevels they have larger receptive fields.  The ability to abstract local patterns along the hierarchyallows better generalizability to unseen cases.

The distinction between C channel/feature and D, dimension are extremely important when it comes to implementing PointNet++. D, dimsion information are propagated via the variable xyz, and C, channel information are propagated via the variable points.

![1](https://github.com/BaiLiping/Radar_PointNet_InstanceSegmentation/blob/main/PointNetPorject_V_0/radar_scenes_description/PointNet++.png)


#### Our Rationale Behind the Parameter Choices:
The parameters governing the partition of field of view is npoints and radius, i.e. how many circles and how big of a circle we use to cover the entire field of view. If the circle is set to be too small, then not enough features would be extracted. If the circles is set to be too large, then there would be too much information condensation and would hinder data processing for later layers.

##### npoints and radius: how many circles and the size of each circle
For Radar Scenes dataset, each sensor has a maximum range detection range of 100m and a field of view of about -60° to +60°. Each frame consists of data collected from all four sensors. A reasonable assumption would be a filed view of 100m*100m.

A reasonable radius for our application might be 5 meters, since within a 5 radius circle, we can capture most features with regard to pedestrians can sedans. For larger vehecles, we would just leave its processing for later layers.

127.3 circles of 5m radius is required to cover the 100m * 100m field of view. Therefore, a reasonable npoints would be 128 for the first layer. 

##### nsamples
The average data points in one frame is 585 points and we assume that the 585 points are uniformly scattered over 100m * 100m area. Therefore, for a sampling circle with 5m radius, there would be 4.59 points scattered within the sampling circle. Therefore, a reasonable nsamples would be 5 for the first layer.

## Dataset Overview

### Website:
[radar-scenes](https://radar-scenes.com/)

### Dataset Composition:
radar scenes consists of three sub-datasets: camera data and odometry data, which are there for data visualization purposes. radar data, which is the input data for our project. There are 158 sequences of data, each consists of multiple scenes. 

### Data Vitualization
the interface provided by the data scenes team are the viewer gui, where radar data are presented alongside odometry and camera data, as shown below:
![1](https://github.com/BaiLiping/Radar_PointNet_InstanceSegmentation/blob/main/PointNetPorject_V_0/radar_scenes_description/radar_scenes_data_visualization.gif)

Notice that the data from the four radars are presented sequentially instead of altogether since the four radars are not synchronized. This is something we need to rectify in order to generate input data for the PointNet.

### Radar Sensor Position

the position of the four radars:
```json
{
  "radar_1": {
    "id": 1,
    "x": 3.663,
    "y": -0.873,
    "yaw": -1.48418552
  },
  "radar_2": {
    "id": 2,
    "x": 3.86,
    "y": -0.7,
    "yaw": -0.436185662
  },
  "radar_3": {
    "id": 3,
    "x": 3.86,
    "y": 0.7,
    "yaw": 0.436
  },
  "radar_4": {
    "id": 4,
    "x": 3.663,
    "y": 0.873,
    "yaw": 1.484
  }
}
```

The positions of the radar and the car coordinate is shown below:
![1](https://github.com/BaiLiping/Radar_PointNet_InstanceSegmentation/blob/main/PointNetPorject_V_0/radar_scenes_description/radar_position.png)

Each radarsensor  independently  determines  the  measurement  timing  onthe basis of internal parameters causing a varying cycle time.On average, the cycle time is60 ms(≈17 Hz).

### Composition of Data

#### Odometry
The **odometry_data** has six columns: 
```
1. timestamp: notice that the frequency of odometry data collection is greater to that of the radars
2. x_seq: x position of the car in the global coordinate
3. y_seq: y position of the car in the global coordinate
4. yaw_seq: yaw direction of the car in the global coordinate
5. vx: the velocity of the ego-vehicle in x-direction
6. yaw_rate: current yaw rate of the car.
```
#### Radar
the **radar_data** has 14 columns:
```
1. timestamp: in micro seconds (10e-6)relative to some arbitrary origin
2. sensor_id: integer value, id of the sensor that recorded the detection
3. range_sc: in meters, radial distance to the detection, sensor coordinate system
4. azimuth_sc: in radians, azimuth angle to the detection, sensor coordinate system
5. rcs: in dBsm, RCS value of the detection
6. vr: in m/s. Radial velocity measured for this detection
7. vr_compensated in m/s: Radial velocity for this detection but compensated for the ego-motion
8. x_cc: in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
9. y_cc: in m, position of the detection in the car-coordinate system (origin is at the center of the rear-axle)
10. x_seq: in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
11. y_seq: in m, position of the detection in the global sequence-coordinate system (origin is at arbitrary start point)
12. uuid: unique identifier for the detection. Can be used for association with predicted labels and debugging
13. track_id: id of the dynamic object this detection belongs to. Empty, if it does not belong to any.
14. label_id: semantic class id of the object to which this detection belongs. passenger cars (0), large vehicles (like agricultural or construction vehicles) (1), trucks (2), busses (3), trains (4), bicycles (5), motorized two-wheeler (6), pedestrians (7), groups of pedestrian (8), animals (9), all other dynamic objects encountered while driving (10), and the static environment (11)
```


### Snippet of Radar Data meta-information

```


here is a code snippet from scenes.json file, there are 4 radars:

```json
  "category": "train",
  "first_timestamp": 156862647501,
  "last_timestamp": 156949185824,
  "scenes": {
    "156862647501": {
      "sensor_id": 1,
      "prev_timestamp": null,
      "next_timestamp": 156862659751,
      "prev_timestamp_same_sensor": null,
      "next_timestamp_same_sensor": 156862719325,
      "odometry_timestamp": 156862651599,
      "odometry_index": 365,
      "image_name": "156862567343.jpg",
      "radar_indices": [
        0,
        3
      ]
    },
    "156862659751": {
      "sensor_id": 2,
      "prev_timestamp": 156862647501,
      "next_timestamp": 156862695773,
      "prev_timestamp_same_sensor": null,
      "next_timestamp_same_sensor": 156862735209,
      "odometry_timestamp": 156862661653,
      "odometry_index": 366,
      "image_name": "156862567343.jpg",
      "radar_indices": [
        3,
        100
      ]
    },
    "156862695773": {
      "sensor_id": 4,
      "prev_timestamp": 156862659751,
      "next_timestamp": 156862701077,
      "prev_timestamp_same_sensor": null,
      "next_timestamp_same_sensor": 156862768707,
      "odometry_timestamp": 156862691606,
      "odometry_index": 369,
      "image_name": "156862567343.jpg",
      "radar_indices": [
        100,
        210
      ]
    },
    "156862701077": {
      "sensor_id": 3,
      "prev_timestamp": 156862695773,
      "next_timestamp": 156862719325,
      "prev_timestamp_same_sensor": null,
      "next_timestamp_same_sensor": 156862773081,
      "odometry_timestamp": 156862701600,
      "odometry_index": 370,
      "image_name": "156862567343.jpg",
      "radar_indices": [
        210,
        311
      ]
    },
```
### Relationship between Timestamps and Scene

```
A scene is defined as one measurement of one of the four radar sensors. For each scene, the sensor id of the respective radar sensor is listed. Each scene has one unique timestamp, namely the time at which the radar sensor performed the measurement. Four timestamps of different radar measurement are given for each scene: the next and previous timestamp of a measurement of the same sensor and the next and previous timestamp of a measurement of any radar sensor. This allows to quickly iterate over measurements from all sensors or over all measurements of a single sensor. For the association with the odometry information, the timestamp of the closest odometry measurement and additionally the index in the odometry table in the hdf5 file where this measurement can be found are given. Furthermore, the filename of the camera image whose timestamp is closest to the radar measurement is given. Finally, the start and end indices of this scene’s radar detections in the hdf5 data set “radar_data” is given. The first index corresponds to the row in the hdf5 data set in which the first detection of this scene can be found. The second index corresponds to the row in the hdf5 data set in which the next scene starts. That is, the detection in this row is the first one that does not belong to the scene anymore. This convention allows to use the common python indexing into lists and arrays, where the second index is exclusive: arr[start:end].
```

### Labels

```
            Label.CAR: ClassificationLabel.CAR,
            Label.LARGE_VEHICLE: ClassificationLabel.LARGE_VEHICLE,
            Label.TRUCK: ClassificationLabel.LARGE_VEHICLE,
            Label.BUS: ClassificationLabel.LARGE_VEHICLE,
            Label.TRAIN: ClassificationLabel.LARGE_VEHICLE,
            Label.BICYCLE: ClassificationLabel.TWO_WHEELER,
            Label.MOTORIZED_TWO_WHEELER: ClassificationLabel.TWO_WHEELER,
            Label.PEDESTRIAN: ClassificationLabel.PEDESTRIAN,
            Label.PEDESTRIAN_GROUP: ClassificationLabel.PEDESTRIAN_GROUP,
            Label.ANIMAL: None,
            Label.OTHER: None,
            Label.STATIC: ClassificationLabel.STATIC

```
# Tips for debugging

Debugging is part of coding. Ideally, one has a clear block diagram for guidance such that the coding process is as fluent as it can be. But even so, there are always details that elude one's mind from time to time. 

One of the most important cue for PyTorch related debugging is the data size. Usuaully it is a layered data structure and would require step by step debugging. 

Set a random seed such that you can recreate the observed bug, and pin point the exact position where the bug shows up. This is a step that would require some trickery and helper functions.  Once the position is located, then start the trial and error process to figure out a solution.
