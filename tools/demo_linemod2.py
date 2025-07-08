from importlib.abc import Loader

from torchvision.io import read_image
from yaml import FullLoader

import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import cv2
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from datasets.linemod2.dataset_demo import PoseDataset as PoseDataset_linemod_demo
from datasets.linemod2.dataset_demo import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import cv2
import numpy as np
from utils.visualization import draw_coordinate_axis, draw_3d_bbox, get_3d_bbox_corners
import matplotlib.pyplot as plt

# dataset_config_dir = '/media/q/SSD2T/1linux/Linemod2_dataset/dataset_config'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/media/q/SSD2T/1linux/Linemod2_dataset/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/media/q/SSD2T/1linux/Linemod2_trained_mod/pose_model_35_0.009038095623254775.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '/media/q/SSD2T/1linux/Linemod2_trained_mod/pose_refine_model_117_0.002372468723333441.pth',  help='resume PoseRefineNet model')
parser.add_argument('--dataset_config_dir', type=str, default = '/media/q/SSD2T/1linux/Linemod2_dataset/dataset_config',  help='model info')
parser.add_argument('--output_result_dir', type=str, default ='/media/q/SSD2T/1linux/Linemod2_eval')
opt = parser.parse_args()

def main():
    # 相机参数
    # 在代码开头添加相机参数加载
    cx = 325.26110
    cy = 242.04899
    fx = 572.41140
    fy = 573.57043
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist_coeffs = np.array(
    #     [camera_data['k1'], camera_data['k2'], camera_data['p1'], camera_data['p2'], camera_data['k3']]) # 相机畸变参数

    # 新增可视化参数
    vis_size = (640, 480)  # 可视化图像尺寸
    axis_length = 0.1  # 坐标系轴长度（米）
    bbox_color = (0, 255, 122)  # 包围框颜色 (BGR)
    bbox_color2 = (0, 0,255)  # 包围框颜色 (BGR)
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # XYZ轴颜色



    #模型的参数
    num_objects = 1
    objlist =[3]
    num_points = 1500
    iteration = 4
    bs = 1

    knn = KNearestNeighbor(1)

    # 模型的加载
    estimator = PoseNet(num_points = num_points, num_obj = num_objects)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
    refiner.cuda()
    refiner.load_state_dict(torch.load(opt.refine_model))
    refiner.eval()

    # 数据集的加载
    testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)
        #损失函数的加载


    sym_list = testdataset.get_sym_list() # 对称物体
    num_points_mesh = testdataset.get_num_points_mesh() # 模型渲染点数量
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    # 缩放直径为米，且缩小10倍
    diameter = []
    meta_file = open('{0}/models_info.yml'.format(opt.dataset_config_dir), 'r')
    meta = yaml.load(meta_file,Loader=FullLoader)
    for obj in objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)


    # 循环加载数据集
    for i, data in enumerate(testdataloader, 0):
        # 解包模型所需数据
        points, choose, img, target, model_points, idx,ori_img = data
        points, choose, img, target, model_points, idx= points.cuda(), \
                                                                 choose.cuda(), \
                                                                 img.cuda(), \
                                                                 target.cuda(), \
                                                                 model_points.cuda(), \
                                                                 idx.cuda()
        # points


        # 预测
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t) # 输出旋转矩阵和平移矩阵


        # 迭代次数
        for ite in range(0, iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_points = torch.bmm((points - T), R).contiguous()
            # 优化
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            # 输出
            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        model_points = model_points[0].cpu().detach().numpy()
        model_points = model_points / 1000.0  # mm -> m

        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()

        pose = np.eye(4)
        pose[:3, :3] = my_r  # 旋转矩阵 (3x3)
        pose[:3, 3] = my_t.squeeze()  # 平移向量 (3,)

        np_img = ori_img[0].cpu().numpy()  # 图像准备
        print("Original shape:", np_img.shape)
        # np_img = np.transpose(np_img, (2,0,1))  # 变成 (H, W, 3)
        np_img = 255 - np_img  # 图像反转（正片恢复）
        np_img = (np_img * 255).astype(np.uint8)  # numpy转为0-255

        # ------------------------- #
        # 可视化坐标轴和包围框绘制部分
        # ------------------------- #
        img_draw = np_img.copy()  # 复制一份图像用于画图（避免原图受修改）
        # 可视化1：绘制坐标系
        draw_coordinate_axis(img_draw, pose, camera_matrix, axis_length, axis_colors)  # 画出坐标系
        # 可视化2：绘制3D包围框
        bbox_corners = get_3d_bbox_corners(model_points) / 1000.0 # 获取gt模型的3D Box
        draw_3d_bbox(img_draw, bbox_corners, pose, camera_matrix, color=bbox_color, thickness=1)  # 画出包围盒
        print("包围盒角点坐标 (单位: 米):\n", bbox_corners)
        print("包围盒尺寸 (X, Y, Z):",
              np.ptp(bbox_corners[:, 0]),  # X轴长度
              np.ptp(bbox_corners[:, 1]),  # Y轴长度
              np.ptp(bbox_corners[:, 2]))  # Z轴长度

        # # 将 model_points 转换到相机坐标系
        # model_points_cam = np.dot(model_points, pose[:3, :3].T) + pose[:3, 3]
        #
        # # 投影到图像平面
        # fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        # cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        #
        # u_list, v_list = [], []
        # for pt in model_points:
        #     x, y, z = pt
        #     if z <= 0:
        #         continue
        #     u = fx * x / z + cx
        #     v = fy * y / z + cy
        #     u_list.append(u)
        #     v_list.append(v)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(img_draw)  # 显示背景图像
        # plt.scatter(u_list, v_list, s=1, c='red', alpha=0.6, label="Model Points")  # 叠加点云投影
        # plt.title("Pose & Point Cloud Visualization")
        # plt.axis('off')
        # plt.show()
        # ------------------------- #
        # 用 Matplotlib 显示最终图像
        # ------------------------- #
        plt.figure(figsize=(8, 6))
        plt.imshow(img_draw)  # BGR -> RGB
        plt.title("Pose Visualization")
        plt.axis('off')
        plt.show()

        #
    #     if idx[0].item() in sym_list:
    #         pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
    #         target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
    #         inds = knn.forward(  knn.k,   target.unsqueeze(0), pred.unsqueeze(0))
    #         target = torch.index_select(target, 1, inds.view(-1) - 1)
    #         dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
    #     else:
    #         dis = np.mean(np.linalg.norm(pred - target, axis=1))
    #
    #     #
    #     if dis < diameter[idx[0].item()]:
    #         success_count[idx[0].item()] += 1
    #         print('No.{0} Pass! Distance: {1}'.format(i, dis))
    #         fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
    #     else:
    #         print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
    #         fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
    #     num_count[idx[0].item()] += 1
    #
    #     #
    # for i in range(num_objects):
    #     print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    #     fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
    # print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    # fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    # fw.close()

if __name__ == '__main__':
    main()
