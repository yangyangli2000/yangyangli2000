
'''
The semantic segmentation of a point should be transformation invariant. 
The usual way to handle this is to align all input sets to a canonical space before feature extraction.

Yet here, the authors of PointNet generate an affine transformation matrix by a mini-network and directly apply this transformation to the coordinates of input points. The mini-network itself resembles the big network and is composed of basic modules of point-independent feature extraction, max pooling, and fully connected layers.

The objective of this k*k T-net is to align features from input. It again takes advantage of the permutation-insensitive function such as max to capture the "transform invariance" of the key features.

Noted that in this T-network idea is depreciated in the later papers published by this group. The
the functionality of PointNet does not seem to hinge on T-net.
'''
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


'''
T-net is just the mini version of the PointNetEncorder. 
It project points into higher dimension first
and then collapse it into a K*K dimension vector, which is then represented by a matrix form.

The idea behind T-net is that if this feature is invariant throughout right transforms, then it would
be captured by the permutation insensative function.
'''
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__() # the input feature has size K*1
        self.conv1 = torch.nn.Conv1d(k, 64, 1) # project from K*1 dimension to 64*1 dimension
        self.conv2 = torch.nn.Conv1d(64, 128, 1) # project from 64*1 dimension to 128*1 dimension
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) # project from 128*1 dimension to 1024*1 dimension
        self.fc1 = nn.Linear(1024, 512) # define first fully connected linear layer with input size 1024*1 and output size 512
        self.fc2 = nn.Linear(512, 256) # define second fully connected linear layer with input size 512*1 and output size 256
        self.fc3 = nn.Linear(256, k * k) # define third fully connected linear layer with input size 256*1 and output size K*K, which is a linear representation of k by k matrix
        self.relu = nn.ReLU() # output layer go through rectified linear unit gating

        self.bn1 = nn.BatchNorm1d(64) #batch normalization for the first convolutional layer
        self.bn2 = nn.BatchNorm1d(128) #batch normalization for the second convolutional layer
        self.bn3 = nn.BatchNorm1d(1024) #batch normalization for the third convolutional layer
        self.bn4 = nn.BatchNorm1d(512) #batch normalization for the first fully connected layer
        self.bn5 = nn.BatchNorm1d(256) #batch normalizationi for the second fully connected layer

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x))) #x transformed by the first layer convolutional network to project into 64 dimension space
        x = F.relu(self.bn2(self.conv2(x))) #x transformed by the second layer convolutional network to project into 128 dimension space
        x = F.relu(self.bn3(self.conv3(x))) #x transformed by the third layer convolutional network to project into 1024 dimension space
        x = torch.max(x, 2, keepdim=True)[0] #maxpooling to collaps the features contained into 1024*1 vector
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x))) # project x from 1024*1 vector into 512*1 vector
        x = F.relu(self.bn5(self.fc2(x))) # project x from 512*1 vector into 256*1 vector
        x = self.fc3(x) # project x from 256*1 vector into (k*k)*1 vector 

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k) # present x in its matrix form, which is a K * K matrix
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=4): #the global_feat flag indicate if it is a classification task or a semantic segmentation task
        super(PointNetEncoder, self).__init__()
        self.stn = STNkd(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1) #the first layer of convolutional network, takes input of dimension channel*1 and output dimension 64*1
        self.conv2 = torch.nn.Conv1d(64, 128, 1) #the second layer of convolutional network, project points from dimension 64 into space with dimension 128
        self.conv3 = torch.nn.Conv1d(128, 1024, 1) #the third layer of convolutional netowrk, project points from dimension of 128 into space with dimension 1024
        self.bn1 = nn.BatchNorm1d(64) #batch norm for the first convolutional network
        self.bn2 = nn.BatchNorm1d(128) #batch norm for the second convolutional network 
        self.bn3 = nn.BatchNorm1d(1024) #batch norm for the third convolutional network
        self.global_feat = global_feat #wheather or not we want to go through the first T-net for input transform
        self.feature_transform = feature_transform #wheather or not we want to go through the second T-net for feature transform.
        if self.feature_transform:
            self.fstn = STNkd(k=64) #the feature transform output a 64*64 matrix

    def forward(self, x):
        B, D, N = x.size() # x[0]= batch_size, x[1]= dimention/channel, x[2]= number of points for this batch
        trans = self.stn(x) #input transform use a T-net that output a channel*channel matrix
        x = x.transpose(2, 1) # exchange position between N and D
        x = torch.bmm(x, trans) # input transform by applying the channel*channel matrix
        x = x.transpose(2, 1) # exchange back N and D, the input transform has completed at this point
        x = F.relu(self.bn1(self.conv1(x))) #go through the first layer of convolutional layer, dimension 64

        if self.feature_transform: #if the feature transform layer is on
            trans_feat = self.fstn(x) #the T-net for feature transform output a 64*64 matrix
            x = x.transpose(2, 1) #exchange position between N and D
            x = torch.bmm(x, trans_feat) # apply the 64*64 feature transform matrix
            x = x.transpose(2, 1) #exchange back N and D, the feature transform has completed at this point
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x))) #go through the second layer of convolutional layer, dimension 128
        x = self.bn3(self.conv3(x)) #go through the third layer of convolutional layer, dimension 1024
        x = torch.max(x, 2, keepdim=True)[0] #max pooling to get a 1024*1 vector
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat #return the 1024*1 vector, the input transform channel*channel matrix and the feature transform 64*64 matrix
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat # if global_feat is False, then go through the semantic segmentation network, need to concatenate the 1024*1 vector with the 64*1 vector to generate a 1088*1 vector 

def feature_transform_reguliarzer(trans):
    d = trans.size()[1] #dimension of the input, channel
    I = torch.eye(d)[None, :, :] #an identity matrix of size channel*channel
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss