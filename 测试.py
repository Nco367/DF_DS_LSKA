from importlib.abc import Loader

from yaml import FullLoader

# import _init_paths
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
from utils.visualization import draw_coordinate_axis, draw_3d_bbox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 虽然这行不直接使用，但有助于3D绘图
import matplotlib.image as mpimg
# dataset_config_dir = '/media/q/SSD2T/1linux/Linemod2_dataset/dataset_config'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/media/q/SSD2T/1linux/Linemod2_dataset/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/media/q/SSD2T/1linux/Linemod2_trained_mod/pose_model_2_0.008679830040346132.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '/media/q/SSD2T/1linux/Linemod2_trained_mod/pose_refine_model_10_0.003791075829140027.pth',  help='resume PoseRefineNet model')
parser.add_argument('--dataset_config_dir', type=str, default = '/media/q/SSD2T/1linux/Linemod2_dataset/dataset_config',  help='model info')
parser.add_argument('--output_result_dir', type=str, default ='/media/q/SSD2T/1linux/Linemod2_eval')
opt = parser.parse_args()

def get_img():

    return

def get_choose():

    return
def get_idx():

    return

def get_point():

    return


def get_3d_bbox_corners(model_points):
    """
    计算点云的轴对齐包围框（AABB）的8个角点
    参数：
        model_points: (N, 3) 形状的NumPy数组，表示点云坐标
    返回：
        bbox_corners: (8, 3) 形状的数组，包含8个角点坐标
    """
    if model_points.ndim == 3:
        model_points = model_points.squeeze(0)  # 去除批次维度
        print("model_points_np shape:", model_points.shape)

    # 1. 计算各轴的最小值和最大值
    min_vals = np.min(model_points, axis=0)  # 形状 (3,)
    max_vals = np.max(model_points, axis=0)  # 形状 (3,)

    # 2. 生成所有可能的极值组合
    x_min, y_min, z_min = min_vals
    x_max, y_max, z_max = max_vals

    # 3. 组合成8个角点
    bbox_corners = np.array([
        [x_min, y_min, z_min],  # 前左下
        [x_max, y_min, z_min],  # 前右下
        [x_max, y_max, z_min],  # 前右上
        [x_min, y_max, z_min],  # 前左上
        [x_min, y_min, z_max],  # 后左下
        [x_max, y_min, z_max],  # 后右下
        [x_max, y_max, z_max],  # 后右上
        [x_min, y_max, z_max]  # 后左上
    ])
    print("model_points_np shape:", bbox_corners.shape)
    return bbox_corners


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
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # XYZ轴颜色



    #模型的参数
    num_objects = 1
    objlist =[3]
    num_points = 2000
    iteration = 4
    bs = 1

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
        diameter.append(meta[obj]['diameter'] / 1000.0)
    print(diameter)


    # 循环加载数据集
    for i, data in enumerate(testdataloader, 0):
        # 解包模型所需数据
        points, choose, img, target, model_points, idx,ori_img = data
        points, choose, img, target, model_points, idx,ori_img = points.cuda(), \
                                                                 choose.cuda(), \
                                                                 img.cuda(), \
                                                         target.cuda(), \
                                                         model_points.cuda(), \
                                                         idx.cuda(), \
                                                          ori_img.cuda()

        ori_img = ori_img.squeeze(0)  # 从 [1,3,480,640] → [3,480,640]
        ori_img_np = ori_img.cpu().detach().numpy()  # 形状为 [H, W, C] 或 [C, H, W]

        if ori_img_np.shape[0] == 3:  # 检查是否为C×H×W格式
            ori_img_np = ori_img_np.transpose(1, 2, 0)  # 转为H×W×C

        if ori_img_np.max() <= 1.0:
            ori_img_np = (ori_img_np * 255).astype(np.uint8)


        # 显示图片
        plt.imshow(ori_img_np)
        plt.axis('off')  # 关闭坐标轴显示
        plt.show()


        print("Shape:", ori_img.shape)  # 应为 (H, W, 3)
        print("Shape:", ori_img_np.shape)  # 应为 (H, W, 3)
        print("Data Type:", ori_img_np.dtype)  # 应为 uint8
        print("Value Range:", ori_img_np.min(), ori_img_np.max())  # 应为 0 和 255


if __name__ == '__main__':
    main()
