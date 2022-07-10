'''
    Function to train PointNet and PointNet++ classification networks by using RadarScenes Dataset.
    RadarScenes Datasets consists of three subsets of data: camera view, which we will use for data presentation purposes;
    Odometry data, which we will use for data presentation purposes; and radar data, which is obtained by 4 radars positioned
    t the front end of a vehicle. For more information regarding the dataset, please refer to README.md
'''
import argparse
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle

import torch
# torch.autograd.set_detect_anomaly(True) # this function is meant to help locate point of failure
torch.manual_seed(5)   # for debugging purpose, set random seed to a specified value
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader

from utils.radar_scenes_dataset_generator import Radar_Scenes_Train_Dataset, Radar_Scenes_Validation_Dataset
from utils.network_validation import validation_metric_for_instance_segmentation, metrics_confusion_matrix, plot_confusion_matrix, get_similarity_distance_threshold
from model.pointnet2_ins_seg import get_pointnet2_for_instance_segmentation_model, get_gmlp_based_pointnet2_for_instance_segmentation_model, get_loss


def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--datapath', default='D:/Tech_Resource/Paper_Resource/Dataset/RadarScenes/RadarScenes/data',
                        type=str, help="dataset main folder")
    parser.add_argument('--numclasses', default=5,  # 6
                        type=int, help='number of classes in the dataset')
    parser.add_argument('--pointCoordDim', default=4, type=int,
                        help='detection point feature dimension, 5 if p=(x_cc,y_cc,vr_compensated, rcs, timestamp)')
    parser.add_argument('--batchsize', default=128,  # change
                        type=int, help='batch size in training')
    parser.add_argument('--num_workers', default=0,
                        type=int, help='number of workers used to load data')
    parser.add_argument('--epoch', default=400,  # change
                        type=int, help='number of epoch in training')
    parser.add_argument('--cuda', default=False,
                        type=bool, help='True to use gpu or False to use cpu')
    parser.add_argument('--gpudevice', default=[0],
                        type=int, help='select gpu devices. Example: [0] or [0,1]', nargs='+')
    parser.add_argument('--train_metric', default=True,
                        type=str, help='whether evaluate on training dataset')
    parser.add_argument('--optimizer', default='ADAM',
                        type=str, help='optimizer for training, SGD or ADAM')
    parser.add_argument('--decay_rate', default=1e-5,  # change
                        type=float, help='decay rate of learning rate for Adam optimizer')
    parser.add_argument('--lr', default=1e-4,  # change
                        type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='CosineAnnealingWarmRestarts', type=str,
                        help='method to update lr. Example: StepLR, CosineAnnealingLR or CosineAnnealingWarmRestarts')
    parser.add_argument('--lr_epoch_half', default=20,  # change
                        type=int, help='(for StepLR) every lr_epoch_half epochs, the lr is reduced by half')
    parser.add_argument('--T_max', default=10, type=int,
                        help='(for CosineAnnealingLR) half of the lr changing period')
    parser.add_argument('--eta_min', default=1e-8, type=float,
                        help='(for CosineAnnealingLR and CosineAnnealingWarmRestarts) minimum learning rate')
    parser.add_argument('--T_0', default=10, type=int,
                        help='(for CosineAnnealingWarmRestarts) number of iterations for the first restart')
    parser.add_argument('--T_mult', default=1, type=int,
                        help='(for CosineAnnealingWarmRestarts) the factor increases T_i after restart')
    parser.add_argument('--model_name', default='pointnet2',
                        type=str, help='pointnet or pointnet2')
    parser.add_argument('--exp_name', default='pointnet2_instance_segmentation\\validation2_instance_segmentation',
                        type=str, help='Name for loading and saving the network in every experiment')
    parser.add_argument('--feature_transform', default=False,
                        type=bool, help="use feature transform in pointnet")
    parser.add_argument('--dataset_D', default=2,
                        type=int, help="the dimension of the coordinate")
    parser.add_argument('--dataset_C', default=2,
                        type=int, help="the dimension of the features/channel")
    parser.add_argument('--first_layer_npoint', default=64,  # 128
                        type=int, help='the number of circles for the first layer')
    parser.add_argument('--first_layer_radius', default=8,  # 5
                        type=int, help='the radius of sampling circles for the first layer')
    parser.add_argument('--first_layer_nsample', default=8,
                        type=int, help='the number of sample for each circle')
    parser.add_argument('--second_layer_npoint', default=16,  # 32
                        type=int, help='the number of circles for the second layer')
    parser.add_argument('--second_layer_radius', default=16,  # 10
                        type=int, help='the radius of sampling circles for the second layer')
    parser.add_argument('--second_layer_nsample', default=8,
                        type=int, help='the number of sample for each circle')
    parser.add_argument('--class_names',
                        default=['CAR', 'PEDESTRIAN', 'PEDESTRIAN_GROUP', 'TWO_WHEELER', 'LARGE_VEHICLE'],  # 'STATIC'
                        type=list, help='a list of class names')
    parser.add_argument('--log_name', default='message.log',
                        type=str, help='the name of log file')
    parser.add_argument('--initial_alpha', default=2,
                        type=float, help='the initial value of alpha')
    parser.add_argument('--epochs_alpha_increase', default=5,
                        type=int, help='every epochs_alpha_increase epochs, alpha will increase')
    parser.add_argument('--alpha_increase', default=2,
                        type=float, help='alpha will increase by alpha_increase every time we train the network for a certain epochs')
    parser.add_argument('--epoch_train_with_Lsim_only', default=5,
                        type=int, help='the first number of epochs we train the network with similarity loss only')
    parser.add_argument('--sample_size', default=100, type=int,
                        help='(used in loading datasets and initialize model) the number of points we want to sample (or repeat) in a frame')
    parser.add_argument('--model_configuration', default='gMLP_based_Pointnet2_for_Instance_Segmentation',
                        type=str, help='Pointnet2_for_Instance_Segmentation or gMLP_based_Pointnet2_for_Instance_Segmentation')
    parser.add_argument('--turn_on_light_weighted_network_model_using_group_conv', default=False, type=bool)
    return parser.parse_args()


def main(args):
    ''' --- SELECT DEVICES --- '''
    device = torch.device("cuda" if args.cuda else "cpu")  # Select either gpu or cpu
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpudevice)  # Select among available GPUs

    '''--- CREATE EXPERIMENTS DIRECTORY AND LOGGERS IN TENSORBOARD --- '''
    projdir = sys.path[0]
    # Path for saving and loading the network.
    saveloadpath = os.path.join(projdir, 'experiment\\checkpoints', args.exp_name + '.pth')
    Path(os.path.dirname(saveloadpath)).mkdir(exist_ok=True, parents=True)
    tblogdir = os.path.join(projdir, 'experiment\\tensorboardX', args.exp_name)
    Path(tblogdir).mkdir(exist_ok=True, parents=True)
    # Create tb_writer(the writer will be used to write the information on tb) by using SummaryWriter,
    tb_writer = SummaryWriter(logdir=tblogdir, flush_secs=3, write_to_disk=True)

    # log_file = open(args.log_name, "w")
    # sys.stdout = log_file

    ''' --- INIT DATASETS AND DATALOADER (FOR SINGLE EPOCH) --- '''
    radar_scenes_transform = None
    radar_scenes_train_dataset = Radar_Scenes_Train_Dataset(args.datapath, radar_scenes_transform,
                                                            sample_size=args.sample_size, LSTM=False, non_static=True)
    radar_scenes_validation_dataset = Radar_Scenes_Validation_Dataset(args.datapath, radar_scenes_transform,
                                                                      sample_size=args.sample_size, LSTM=False, non_static=True)
    # Create dataloader for training by using batch_size frames' data in each batch
    trainDataLoader = DataLoader(radar_scenes_train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    validationDataLoader = DataLoader(radar_scenes_validation_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    print("Training Data Successfully Loaded to Its Batch Form")
    print("-" * 120)

    ''' --- INIT NETWORK MODEL --- '''
    if args.model_configuration == 'Pointnet2_for_Instance_Segmentation':
        instance_segmentor = get_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                              args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                              args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample,
                                              args.turn_on_light_weighted_network_model_using_group_conv)
    elif args.model_configuration == 'gMLP_based_Pointnet2_for_Instance_Segmentation':
        instance_segmentor = get_gmlp_based_pointnet2_for_instance_segmentation_model(args.numclasses, args.dataset_D, args.dataset_C,
                                              args.first_layer_npoint, args.first_layer_radius, args.first_layer_nsample,
                                              args.second_layer_npoint, args.second_layer_radius, args.second_layer_nsample, args.sample_size,
                                              args.turn_on_light_weighted_network_model_using_group_conv)
    instance_segmentor = instance_segmentor.to(device)

    ''' --- LOAD NETWORK IF EXISTS --- '''
    if os.path.exists(saveloadpath):
        print('Using pretrained model found...')
        print("-" * 120)
        checkpoint = torch.load(saveloadpath)
        # Counting starts from 1 rather than 0 when print the information of start_epoch
        start_epoch = checkpoint['epoch'] + 1  # 从上次训练的下一个epoch开始
        iteration = checkpoint['iteration']
        best_validation_mmCov = checkpoint['validation_mmCov']
        best_validation_mmAP = checkpoint['validation_mmAP']
        instance_segmentor.load_state_dict(checkpoint['last_epoch_model_state_dict'])
        best_model_state_dict = checkpoint['best_model_state_dict']
        best_model_Ths = checkpoint['best_model_Ths']
        alpha = checkpoint['last_epoch_alpha']
        epoch_train_with_Lsim_only = checkpoint['epoch_train_with_Lsim_only']

    else:
        print('No existing model, starting training from scratch...')
        print("-" * 120)
        # Counting starts from 1 rather than 0 when print the information of start_epoch
        start_epoch = 1
        iteration = 1  # Counting starts from 1 rather than 0 when print the information of iteration
        best_validation_mmCov = 0
        best_validation_mmAP = 0
        best_model_state_dict = instance_segmentor.state_dict()  # 将初始模型设置为best_model_state_dict
        best_model_Ths = np.ones(args.numclasses)
        alpha = args.initial_alpha
        epoch_train_with_Lsim_only = args.epoch_train_with_Lsim_only  # 前epoch_train_with_Lsim_only个epoch只使用L_sim作为损失函数

    ''' --- CREATE OPTIMIZER ---'''
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(instance_segmentor.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(instance_segmentor.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.decay_rate)

    ''' --- CHOOSE LR SCHEDULER --- '''
    if args.lr_scheduler == 'StepLR':
        # half(0.5) the learning rate every 'step_size' epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epoch_half, gamma=0.5)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

    # log info
    printparams = 'Model parameters:' + json.dumps(vars(args), indent=4, sort_keys=True)
    print(printparams)
    print("-" * 120)
    tb_writer.add_text('hyper-parameters', printparams, iteration)  # tb_writer.add_hparam(args)

    ''' --- START TRAINING ---'''
    for epoch in range(start_epoch, args.epoch + 1):
        print('Epoch %d/%s:' % (epoch, args.epoch))

        instance_segmentor = instance_segmentor.train()  # Sets the module in training mode

        # Add the "learning rate" into tensorboard scalar which will be shown in tensorboard
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iteration)

        for batch_id, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9, desc='Training'):
            points, label = data  # the data has the form (B, N, C) the label has the form (B, 2, N)
            B, _, _ = points.shape
            points = points.permute(0, 2, 1)  # change it into the form (B, C, N)
            points, label = points.float().to(device), label.float().to(device)
            optimizer.zero_grad()  # Reset gradients
            pred = instance_segmentor(points, args.dataset_D)  # Forward propagation
            loss_calculator = get_loss()
            sim_loss, cf_loss_for_similarity, sem_loss = loss_calculator(pred, label, device, alpha=alpha, margin=[1., 2.])  # calculate loss
            if epoch <= epoch_train_with_Lsim_only:
                loss = sim_loss
            else:
                loss = sim_loss + cf_loss_for_similarity + sem_loss
            loss.backward()  # Back propagation
            optimizer.step()  # Update all Variables by using newly fetched gradients
            if batch_id % 10 == 0 or batch_id >= len(trainDataLoader)-1:  # 每10个batch输出一次loss；最后1个batch也输出
                print('Training loss is {}'.format(loss.item()))
            if not batch_id % 5:
                # Log once for every 5 batches, add the "train_loss/cross_entropy" into tensorboard scalar which will be shown in tensorboard
                tb_writer.add_scalar('train_loss/cross_entropy', loss.item(), iteration)
            iteration += 1
        scheduler.step()
        if epoch % args.epochs_alpha_increase == 0:
            alpha += args.alpha_increase  # 每epochs_alpha_increase个epoch, alpha会增加alpha_increase

        ''' --- VALIDATION AND SAVE NETWORK --- '''
        if not epoch % 1:  # Doing the following things every epoch.
            Th_s = get_similarity_distance_threshold(instance_segmentor, trainDataLoader, args.dataset_D, args.numclasses, device)
            # Th_s = np.array([1, 1, 1, 1, 1])
            # Perform predictions on the training data and calculate the metrics.
            train_mmCov, train_mmAP = validation_metric_for_instance_segmentation(instance_segmentor, trainDataLoader,
                                                                                  args.dataset_D, device, Th_s)
            # Perform predictions on the validation data and calculate the metrics.
            validation_mmCov, validation_mmAP = validation_metric_for_instance_segmentation(instance_segmentor,
                                                                                            validationDataLoader, args.dataset_D, device, Th_s)

            print('Train mmCov: {}\nvalidation mmCov: {}'.format(train_mmCov, validation_mmCov))
            print('Train mmAP: {}\nvalidation mmAP: {}'.format(train_mmAP, validation_mmAP))

            '''
            B_validation = validation_label.shape[0]
            # Add the "train_mmAP" "validation_mmAP" into tensorboard scalars which will be shown in tensorboard.
            tb_writer.add_scalars(
                'metrics/mmAP', {'train': train_mmAP, 'validation': validation_mmAP}, iteration)
            for batch_index in range(B_validation):
                # Calculate confusion matrix.
                confmatrix_validation = metrics_confusion_matrix(validation_label[batch_index], validation_pred[batch_index])
                print('validation confusion matrix: \n', confmatrix_validation)
                # Log confusion matrix.
                fig, ax = plot_confusion_matrix(confmatrix_validation, args.class_names, normalize=False,
                                                title='validation Confusion Matrix')
                # Log normalized confusion matrix.
                fig_n, ax_n = plot_confusion_matrix(confmatrix_validation, args.class_names, normalize=True,
                                                    title='validation Confusion Matrix - Normalized')
                # Add the "confusion matrix" "normalized confusion matrix" into tensorboard figure which will be shown in tensorboard.
                tb_writer.add_figure('validation_confusion_matrix/abs', fig, global_step=iteration, close=True)
                tb_writer.add_figure('validation_confusion_matrix/norm', fig_n, global_step=iteration, close=True)

            # Log precision recall curves.
            for batch_index in range(B_validation):
                for idx, clsname in enumerate(args.class_names):
                    # Convert log_softmax to softmax(which is actual probability) and select the desired class.
                    validation_pred_binary = torch.exp(validation_pred[batch_index][:, idx])
                    validation_targ_binary = validation_label[batch_index].eq(idx)
                    # Add the "precision recall curves" which will be shown in tensorboard.
                    tb_writer.add_pr_curve(tag='pr_curves/' + clsname, labels=validation_targ_binary,
                                           predictions=validation_pred_binary, global_step=iteration)
            '''
            ''' --- SAVE NETWORK --- '''

            best_validation_mmCov = validation_mmCov if validation_mmCov > best_validation_mmCov else best_validation_mmCov
            best_model_state_dict = instance_segmentor.state_dict() if validation_mmAP > best_validation_mmAP else best_model_state_dict
            best_model_Ths = Th_s if validation_mmAP > best_validation_mmAP else best_model_Ths
            best_validation_mmAP = validation_mmAP if validation_mmAP > best_validation_mmAP else best_validation_mmAP
            state = {
                'epoch': epoch,
                'iteration': iteration,
                'validation_mmCov': best_validation_mmCov,
                'validation_mmAP': best_validation_mmAP,
                'last_epoch_model_state_dict': instance_segmentor.state_dict(),
                'best_model_state_dict': best_model_state_dict,
                'best_model_Ths': best_model_Ths,
                'last_epoch_optimizer_state_dict': optimizer.state_dict(),
                'last_epoch_alpha': alpha,
                'epoch_train_with_Lsim_only': epoch_train_with_Lsim_only,
            }
            torch.save(state, saveloadpath)
            print('Model saved!!!')

    print('Best mmCov: %f' % best_validation_mmCov)
    print('Best mmAP: %f' % best_validation_mmAP)

    tb_writer.close()
    # log_file.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
