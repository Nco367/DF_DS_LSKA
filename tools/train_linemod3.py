# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
from torch import utils
from torch import manual_seed, load
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
# from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.linemod3.dataset import PoseDataset as PoseDataset_linemod3
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils.utils import setup_logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'linemod', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 1, help='batch size')
parser.add_argument('--workers', type=int, default = 4, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='DS_LSKA_0915_pose_model_3_0.009499964444953412.pth')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='DS_LSKA_0915_pose_refine_model_37_0.00676745064561135.pth')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
parser.add_argument('--k', type=int, default = 20, help='knn')
opt = parser.parse_args()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_point_clouds(points1, points2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points1[:,0], points1[:,1], points1[:,2], s=1, c='r', label="model_points (pred)")
    ax.scatter(points2[:,0], points2[:,1], points2[:,2], s=1, c='b', label="target (GT)")
    ax.legend()
    plt.show()

def main():
# 随机数设置 ?
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed) # 设置 random 模块的随机数生成器种子为 opt.manualSeed
    manual_seed(opt.manualSeed) #  设置 PyTorch 中的随机数生成器的种子
# 数据集参数设置
    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = '/media/q/SSD2T/1linux/YCB_trained_models' #folder to save trained models
        opt.log_dir = '/media/q/SSD2T/1linux/YCB_log' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 3
        opt.num_points = 700
        opt.outf = '/media/q/SSD2T/1linux/Linemod3_trained_mod' # 设置训练过程中保存模型的文件夹路径
        opt.log_dir = '/media/q/SSD2T/1linux/Linemod3_log' # 设置训练过程中日志保存的目录
        opt.repeat_epoch = 1 # 设置每个物体训练的重复轮数为 20
    else:
        print('Unknown dataset')
        return



# —————————————————————————————— 训练、优化网络实例化——————————————————————————————

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

# 网络间断点设置
    if opt.resume_posenet != '':
        estimator.load_state_dict(load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
    if opt.resume_refinenet != '':
        refiner.load_state_dict(load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size =max( 1,int(opt.batch_size / opt.iteration))
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

# 数据加载
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod3('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod3('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

# 对称信息获取
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh() # 数据集中网格模型的点数

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

#—————————————————————————————— 损失函数——————————————————————————————
    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

# 测试设置
    best_test = np.inf

    # 清空日志
    # 创建带时间戳的日志目录
    st_time = time.time()
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(st_time))
    log_dir = os.path.join(opt.log_dir, time_str)

    # 如果目录不存在，就创建
    os.makedirs(log_dir, exist_ok=True)
    print(f"本次训练日志保存到: {log_dir}")

    for epoch in range(opt.start_epoch, opt.nepoch):
        # 日志
        logger = setup_logger('epoch%d' % epoch, os.path.join(log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        # train的flag
        train_count = 0
        train_dis_avg = 0.0

        # 是否训练优化
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad() # 清除上一轮计算的梯度
        # Estimator 参数量
        estimator_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
        print("Estimator Params:", estimator_params)

        # Refiner 参数量
        refiner_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
        print("Refiner Params:", refiner_params)
        # 重复训练轮次
        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data
                points, choose, img, target, model_points, idx = points.cuda(), \
                                                                 choose.cuda(), \
                                                                 img.cuda(), \
                                                                 target.cuda(), \
                                                                 model_points.cuda(), \
                                                                 idx.cuda()

                # points：深度掩码后的点云
                # choose：一个索引数组，指示哪些点被选中，用于训练或评估。
                # img：图像数据，经过归一化的RGB图像。
                # target：目标的 3D 点云坐标，经过变换后的物体模型点云数据。
                # model_points：物体模型的 3D 点云数据。
                # idx：物体的索引，通常是一个整数，表示该样本所属的物体类别。
                # ——————————————————————————————————————位姿估计——————————————————————————————————————————
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                # ——————————————————————损失函数——————————————————————————
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w,opt.refine_start, opt.num_points_mesh, opt.sym_list)
                #——————————————————————————优化——————————————————————————
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                        # scaler.scale(dis).backward()
                else:
                    loss.backward()
                    # scaler.scale(loss).backward()

                # torch.nn.utils.clip_grad_norm_(estimator.parameters(), max_norm=1.0)
                # dis是预测点和目标点的距离，
                train_dis_avg += dis.item()
                train_count += 1
                # print("pred_t:", pred_t[0].detach().cpu().numpy())
                # print("pred_r norm:",  pred_r[0].detach().cpu().numpy())
                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad() # 清除上一轮计算的梯度
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = points.cuda(), \
                                                             choose.cuda(), \
                                                             img.cuda(), \
                                                             target.cuda(), \
                                                             model_points.cuda(), \
                                                             idx.cuda()

            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start, opt.num_points_mesh, opt.sym_list)

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))


            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        logger.info("pred_r range: [{:.4f}, {:.4f}], pred_t range: [{:.4f}, {:.4f}]".format(
            pred_r.min().item(), pred_r.max().item(),
            pred_t.min().item(), pred_t.max().item()))
        if j == 0:  # 只看第一个测试样本
            # 假设 model_points, target 是 torch.Tensor
            plot_point_clouds(new_points.cpu().numpy(), new_target.cpu().numpy())

            break

        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        # 调节衰减率
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        # 调节衰减率
        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = max(1, int(opt.batch_size / opt.iteration))
            # opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)


            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod3('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod3('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()
