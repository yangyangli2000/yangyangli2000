'''
    Function to train PointNet and PointNet++ classification networks by using Radar Scenes Dataset.
    Radar Scenes Datasets consists of three subset of datas, camera view, which we will use for data presentation purposes.
    Odometry data, which we will use for data presentation purposes, and radar data, which is obtained by 4 radars positioned 
    t the front end of a vehicle. For more information regarding the dataset, please refer to README.md   
'''
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle

import torch
#torch.autograd.set_detect_anomaly(True) # this function is meant to help locate point of failure
#torch.manual_seed(0)   # for debugging purpose, set random seed to a speficied value
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader

from utils.radar_scenes_dataset_generator import radar_scenes_partitioned_data_generator, Radar_Scenes_Train_Dataset, Radar_Scenes_Validation_Dataset
from utils.network_validation import validation, metrics_accuracy, metrics_confusion_matrix, plot_confusion_matrix
from model.pointnet2_sem_seg import get_model, get_loss


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    # parser.add_argument('--datapath',           default='D:/Tech_Resource/Paper_Resource/Dataset/RadarScenes/RadarScenes/radar_scenes_processed_data',
    #                                                                        type=str,   help="dataset main folder")
    parser.add_argument('--datapath',           default='/home/blp/Desktop/radar_scenes/data',
                        type=str,   help="dataset main folder")
    parser.add_argument('--numclasses',         default=6,
                        type=int,   help='number of classes in the dataset')
    parser.add_argument('--pointCoordDim',      default=4,                  type=int,
                        help='detection point feature dimension, 5 if p=(x_cc,y_cc,vr_compensated, rcs, timestamp)')
    parser.add_argument('--batchsize',          default=64,
                        type=int,   help='batch size in training')
    parser.add_argument('--num_workers',        default=0,
                        type=int,   help='number of workers used to load data')
    parser.add_argument('--epoch',              default=400,
                        type=int,   help='number of epoch in training')
    parser.add_argument('--cuda',               default=True,
                        type=bool,  help='True to use gpu or False to use cpu')
    parser.add_argument('--gpudevice',          default=[
                        0],              type=int,   help='select gpu devices. Example: [0] or [0,1]', nargs='+')
    parser.add_argument('--train_metric',       default=True,
                        type=str,   help='whether evaluate on training dataset')
    parser.add_argument('--optimizer',          default='ADAM',
                        type=str,   help='optimizer for training, SGD or ADAM')
    parser.add_argument('--decay_rate',         default=1e-5,
                        type=float, help='decay rate of learning rate for Adam optimizer')
    parser.add_argument('--lr',                 default=1e-4,
                        type=float, help='learning rate')
    parser.add_argument('--lr_epoch_half',      default=20,                 type=int,
                        help='every lr_epoch_half epochs, the lr is reduced by half')
    parser.add_argument('--model_name',         default='pointnet2',
                        type=str,   help='pointnet or pointnet2')
    parser.add_argument('--exp_name',           default='pointnet2\\validation2',
                        type=str,   help='Name for loading and saving the network in every experiment')
    parser.add_argument('--feature_transform',  default=False,
                        type=bool,  help="use feature transform in pointnet")
    parser.add_argument('--dataset_D',          default=2,
                        type=int,   help="the dimension of the coordinate")
    parser.add_argument('--dataset_C',          default=2,
                        type=int,  help="the dimension of the features/channel")
    parser.add_argument('--first_layer_npoint', default=128,
                        type=int,   help='the number of circles for the first layer')
    parser.add_argument('--first_layer_radius', default=5,
                        type=int,   help='the radius of sampling circles for the first layer')
    parser.add_argument('--first_layer_nsample', default=8,
                        type=int,   help='the number of sample for each circle')
    parser.add_argument('--second_layer_npoint', default=32,
                        type=int,   help='the number of circles for the second layer')
    parser.add_argument('--second_layer_radius', default=10,
                        type=int,   help='the radius of sampling circles for the second layer')
    parser.add_argument('--second_layer_nsample', default=8,
                        type=int,   help='the number of sample for each circle')
    parser.add_argument('--class_names', default=['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE', 'STATIC'],
                        type=list,   help='a list of class names')
    return parser.parse_args()


def main(args):
    ''' --- SELECT DEVICES --- '''
    device = torch.device("cuda" if args.cuda else "cpu")  # Select either gpu or cpu
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpudevice)  # Select among available GPUs
    ''' --- INIT DATASETS AND DATALOADER (FOR SINGLE EPOCH) --- '''
    radar_scenes_transform = None
    radar_scenes_train_dataset = Radar_Scenes_Train_Dataset(
        args.datapath, radar_scenes_transform, sample_size=500, LSTM=False, non_static=False)
    radar_scenes_validation_dataset = Radar_Scenes_Validation_Dataset(
        args.datapath, radar_scenes_transform, sample_size=500, LSTM=False, non_static=False)
    # Create dataloader for training by using batch_size frames' data in each batch
    trainDataLoader = DataLoader(
        radar_scenes_train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    print("Training Data Successfully Loaded to Its Batch Form")
    print("-" * 120)

    ''' --- INIT NETWORK MODEL --- '''
    classifier = get_model(args.numclasses,
                           args.dataset_D,
                           args.dataset_C,
                           args.first_layer_npoint,
                           args.first_layer_radius,
                           args.first_layer_nsample,
                           args.second_layer_npoint,
                           args.second_layer_radius,
                           args.second_layer_nsample
                           )
    classifier = classifier.to(device)

    ''' --- LOAD NETWORK IF EXISTS --- '''
    projdir = sys.path[0]
    # Path for saving and loading the network.
    saveloadpath = os.path.join(projdir, 'experiment\\checkpoints', args.exp_name+'.pth')
    Path(os.path.dirname(saveloadpath)).mkdir(exist_ok=True, parents=True)
    tblogdir = os.path.join(projdir, 'experiment\\tensorboardX', args.exp_name)
    Path(tblogdir).mkdir(exist_ok=True, parents=True)
    # Create tb_writer(the writer will be used to write the information on tb) by using SummaryWriter,
    tb_writer = SummaryWriter(logdir=tblogdir, flush_secs=3, write_to_disk=True)

    print('Using pretrained model found...')
    print("-" * 120)
    classifier = torch.load(saveloadpath)
    iteration = 1  # Counting starts from 1 rather than 0 when print the information of iteration
    best_validation_acc = 0



    ''' --- VALIDATION AND SAVE NETWORK --- '''
    # Perform predictions on the training data.
    train_targ, train_pred = validation(classifier, radar_scenes_train_dataset, args.dataset_D, device, num_workers=args.num_workers, batch_size=args.batchsize)
    B_train, _ = train_targ.shape
    # Perform predictions on the validationing data.
    validation_targ,  validation_pred = validation(classifier, radar_scenes_validation_dataset, args.dataset_D, device,  num_workers=args.num_workers, batch_size=args.batchsize)
    B_validation, _= validation_targ.shape
    # Calculate the accuracy rate for training data.
    train_acc = 0 
    for batch_index in range(B_train):
        train_acc += metrics_accuracy(train_targ[batch_index],
                                      train_pred[batch_index])

    # Calculate the accuracy rate for validationing data.
    validation_acc = 0
    for batch_index in range(B_validation):
        validation_acc += metrics_accuracy(train_targ[batch_index],
                                      train_pred[batch_index])

    print('Train Accuracy: {}\nvalidation Accuracy: {}'.format(train_acc, validation_acc))
    # Add the "train_acc" "validation_acc" into tensorboard scalars which will be shown in tensorboard.
    tb_writer.add_scalars('metrics/accuracy', {'train': train_acc, 'validation': validation_acc}, iteration)
    for batch_index in range(B_validation):
        # Calculate confusion matrix.
        confmatrix_validation = metrics_confusion_matrix(
            validation_targ[batch_index], validation_pred[batch_index])
        print('validation confusion matrix: \n', confmatrix_validation)
        # Log confusion matrix.
        fig,ax = plot_confusion_matrix(confmatrix_validation, args.class_names, normalize=False, title='validation Confusion Matrix')
        # Log normalized confusion matrix.
        fig_n,ax_n = plot_confusion_matrix(confmatrix_validation, args.class_names, normalize=True,  title='validation Confusion Matrix - Normalized')
        # Add the "confusion matrix" "normalized confusion matrix" into tensorboard figure which will be shown in tensorboard.
        tb_writer.add_figure('validation_confusion_matrix/abs',  fig,   global_step=iteration, close=True)
        tb_writer.add_figure('validation_confusion_matrix/norm', fig_n, global_step=iteration, close=True)

    # Log precision recall curves.
    for batch_index in range(B_validation):
        for idx, clsname in enumerate(args.class_names):
            # Convert log_softmax to softmax(which is actual probability) and select the desired class.
            validation_pred_binary = torch.exp(
                validation_pred[batch_index][:, idx])
            validation_targ_binary = validation_targ[batch_index].eq(idx)
            # Add the "precision recall curves" which will be shown in tensorboard.
            tb_writer.add_pr_curve(tag='pr_curves/'+clsname, labels=validation_targ_binary,predictions=validation_pred_binary, global_step=iteration)

    tb_writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
