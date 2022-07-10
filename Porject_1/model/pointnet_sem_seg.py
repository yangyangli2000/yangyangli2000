'''
The following code is adapted from the latest implementation of PointNet with Pytorch
https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The purpose of this network is to generate Semantic Segmentation from a point cloud, which is formulated as
a per-point classification problem. The detailed problem setup, please refer to section 5.1.3 Semantic Segmentation in Scenes
in the paper: https://arxiv.org/pdf/1612.00593.pdf

Semantic Segmentation is an extension of the classification task. In the original paper, the procedure for data processing is the following:
We uniformly sample 1024 points on mesh faces according to the face area and normalize them into a unit sphere.
During training, we augment the point cloud on the fly by randomly rotating the object along the up-axis and jitter the position of each points by Gaussian noise with zero mean and 0.02 standard deviation

However, for our radar scenes dataset,  we choose to omit the Normalization step and only implement the sampling and data augmentation steps, like the following:

1. Sample N points from each frame after augmentation. Please note the nomenclature that channel means the information contained in each point.
2. Project the points into higher dimension spaces via Neural Networks. 
3. Apply a permutation insensitive function, in this case, max-pooling to each point such that the global feature is represented by a 1024*1 vector.
4. For classification tasks, all we need to do is to push this 1024*1 vector through a network as we would a regular image and obtain the classification probability vector.
5. For semantic segmentation tasks, we need to combine each point with this global vector and generate a classification probability vector for each point.

For a block diagram view of this process, please refer to the aforementioned paper, Figure 2.

Noted that there are two paths, as flagged by "global_feat". If the global_feat is true, the network goes through
the classification pathway of Figure 2. Otherwise, the network goes through the semantic segmentation pathway of Figure 2.
'''
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from model.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_class):
        super(get_model, self).__init__()
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=4) #global_feat is false, in order to go through the semantic segmentation path instead of the classification path, the output is a 1088*1 vector
        self.conv1 = torch.nn.Conv1d(1088, 512, 1) #the input size is 1088*1, output size 512*1
        self.conv2 = torch.nn.Conv1d(512, 256, 1) #the second layer of convolutional network takes input 512*1 and output size 256*1
        self.conv3 = torch.nn.Conv1d(256, 128, 1) #the third layer of convolutional netowrk takes input 256*1 and output size 128*1
        self.conv4 = torch.nn.Conv1d(128, self.k, 1) #the fouth layer of convolutional network takes input 128*1 and output channel*1
        self.bn1 = nn.BatchNorm1d(512) #the batchnorm for first layer
        self.bn2 = nn.BatchNorm1d(256) #the batchnorm for second layer
        self.bn3 = nn.BatchNorm1d(128) #the batchnorm for third layer

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x) #points go through PointNetEncoder to generate 1088*1 vector
        x = F.relu(self.bn1(self.conv1(x))) #points go through the first convolutional layer, 512*1
        x = F.relu(self.bn2(self.conv2(x))) #points go through the second convolutional layer, 256*1
        x = F.relu(self.bn3(self.conv3(x))) #poitns go through the third convolutional layer, 128*1
        x = self.conv4(x) #points go through the forth convolutional layer, channel*1
        x = x.transpose(2,1).contiguous() #exchange position between N and C
        x = F.log_softmax(x.view(-1,self.k), dim=-1) #go through the softmax to generate class probability of each point
        x = x.view(batchsize, n_pts, self.k)
        return x, trans_feat # return the class probability(semantic segmentation) of each point classification

class get_loss(torch.nn.Module): #compute the loss for the predictions
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight= weight) #the loss for the main network
        mat_diff_loss = feature_transform_reguliarzer(trans_feat) #the loss for the feature transform matrix
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale #the total loss if the combination of the two
        return total_loss #return the total loss

    def backward(self,pred,target):
        delta = F.nll_loss(pred, target.long()).backward()


if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(12, 3, 2048)
    (model(xyz))