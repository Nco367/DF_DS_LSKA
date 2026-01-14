from importlib.abc import Loader
from sys import prefix

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
from datasets.linemod5_eval.dataset import PoseDataset as PoseDataset_linemod5
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import matplotlib.pyplot as plt
from utils.visualization import draw_coordinate_axis, draw_3d_bbox, get_3d_obb_corners
import cv2
import time

from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/media/q/SSD2T/1linux/Linemod5_dataset/Linemod_preprocessed', help='dataset root dir')
parser.add_argument('--model', type=str, default = '/media/q/SSD2T/1linux/Linemod5_trained_mod/pose_model_3_0.007043762697527806.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '/media/q/SSD2T/1linux/Linemod5_trained_mod/pose_refine_model_5_0.005734924412487696.pth',  help='resume PoseRefineNet model')
opt = parser.parse_args()

cx = 2038.63190
cy = 1408.83467
fx = 5070.07988
fy = 5142.08286

camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist_coeffs = np.array(
    # [camera_data['k1'], camera_data['k2'], camera_data['p1'], camera_data['p2'], camera_data['k3']]) # 相机畸变参数
    # 新增可视化参数
vis_size = (4000, 2700)  # 可视化图像尺寸
axis_length = 0.1  # 坐标系轴长度（米）
bbox_color = (0, 255, 122)  # 包围框颜色 (BGR)
bbox_color2 = (255, 0,0)  # 包围框颜色 (BGR)
axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # XYZ轴颜色

#模型的参数
num_objects = 1
objlist = [3]
num_points = 1000
iteration =2
bs = 1

dataset_config_dir = '/media/q/SSD2T/1linux/Linemod5_dataset/dataset_config'
output_result_dir = '/media/q/SSD2T/1linux/Linemod5_eval'
knn = KNearestNeighbor(1)

# 网络和模型加载
estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()

estimator.load_state_dict(torch.load(opt.model), strict=True)
refiner.load_state_dict(torch.load(opt.refine_model), strict=True)

estimator.eval()
refiner.eval()
# Estimator 参数量
estimator_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
print("Estimator Params:", estimator_params)

# Refiner 参数量
refiner_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
print("Refiner Params:", refiner_params)

# 数据加载器和评估器
testdataset = PoseDataset_linemod5('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file, Loader=FullLoader)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.01)
print(diameter)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

# 基础目录
base_dir = "/media/q/SSD2T/1linux/Linemod5_eval/evaluation_result/"

# 生成时间戳，比如：2025-09-14_14-55-32
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 在 base_dir 下面新建一个时间戳子目录
save_dir = os.path.join(base_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

print(f"结果保存目录: {save_dir}")
import time

total_frames = len(testdataset)
start_time = time.time()

print('Start inference...')
for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx , ori_img = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, target, model_points, idx ,ori_img = points.cuda(), \
                                                     choose.cuda(), \
                                                     img.cuda(), \
                                                     target.cuda(), \
                                                     model_points.cuda(), \
                                                     idx.cuda(), \
                                                     ori_img.cuda()

    # 🔹 反查原始 RGB 路径______________________________________________________________________
    rgb_path = testdataset.list_rgb[i]
    frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
    print(f"当前样本原始帧号: {frame_id}")

    torch.cuda.empty_cache()
    with torch.no_grad():
        # 第一步：PoseNet 推理（estimator）
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1) # 对 pred_r 在第 2 维上计算 L2 范数，并将其归一化，确保旋转表示保持单位长度，
        pred_c = pred_c.view(bs, num_points)  # 将置信度 pred_c 重塑为形状为 (bs, num_points) 的二维张量
        how_max, which_max = torch.max(pred_c, 1) # 找出每个批次中每个样本的最大置信度值（how_max）及其对应的索引（which_max）
        pred_t = pred_t.view(bs * num_points, 1, 3) # 将平移预测 pred_t 重塑为形状为 (bs * num_points, 1, 3) 的张量

        # 第二步：refiner 迭代细化
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy() # 展平
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy() # 展平
        my_pred = np.append(my_r, my_t) # 组合
        for ite in range(0, iteration):
            T = torch.from_numpy(my_t.astype(np.float32)).cuda().view(1, 3).repeat(num_points,1).contiguous().view(1,num_points,3) # 平移矩阵移动到gpu
            my_mat = quaternion_matrix(my_r) # 旋转四元数转矩阵
            R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).cuda().view(1, 3, 3) # 旋转转gpu
            my_mat[0:3, 3] = my_t # 平移加到矩阵

            new_points = torch.bmm((points - T), R).contiguous() # 经过偏移后的新点云
            pred_r, pred_t = refiner(new_points, emb, idx) # 细化网络
            pred_r = pred_r.view(1, 1, -1) # 改tenser形状
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1)) #  归一化 处理
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy() # 这两个张量转换为 NumPy 数组
            my_mat_2 = quaternion_matrix(my_r_2) # # 旋转四元数转矩阵
            my_mat_2[0:3, 3] = my_t_2 # 平移加到矩阵

            my_mat_final = np.dot(my_mat, my_mat_2) # 矩阵乘积
            my_r_final = copy.deepcopy(my_mat_final) # 深拷贝
            my_r_final[0:3, 3] = 0 # 将 my_r_final 矩阵的前 3 行（第 0 行到第 2 行）中的第 3 列的所有元素设为 0
            my_r_final = quaternion_from_matrix(my_r_final, True) # 将变换矩阵 my_r_final 从 4x4 矩阵转换为四元数表示
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]]) # 从 my_mat_final 中提取平移部分

            my_pred = np.append(my_r_final, my_t_final) # 将旋转四元数 my_r_final 和位移向量 my_t_final 合并为一个数组 my_pred
            my_r = my_r_final
            my_t = my_t_final

     #Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        # 第三步：后处理（刚体变换、计算 dis 等）
        model_points = model_points[0].cpu().detach().numpy() # 从 PyTorch 张量转换为 NumPy 数组
        my_r = quaternion_matrix(my_r)[:3, :3] # 将四元数 my_r 转换为旋转矩阵，并提取出旋转矩阵的前三行前三列
        pred = np.dot(model_points, my_r.T) + my_t
        # model_points 是点云数据
        # my_r.T 是旋转矩阵 my_r 的转置
        # my_t 是平移向量
        # 对model_points 执行 刚性变换
        target = target[0].cpu().detach().numpy() # 将一个张量 target 从 PyTorch 中的计算图中分离出来，并将其转换为 NumPy 数组

        if idx[0].item() in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            # 将一个 NumPy 数组 pred 转换为 PyTorch 张量，并做一些张量操作，最终得到一个适合 GPU 计算的张量
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            # 将 NumPy 数组 target 转换为 PyTorch 张量
            inds = knn.forward(knn.k, target.unsqueeze(0), pred.unsqueeze(0))
            # 来执行 k 最近邻（KNN）操作；k用于指定每次查询时返回多少个邻居
            target = torch.index_select(target, 1, inds.view(-1) - 1)

            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()

        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))

        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            print('No.{0} Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} Pass! Distance: {1}\n'.format(i, dis))
        else:
            print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
            fw.write('No.{0} NOT Pass! Distance: {1}\n'.format(i, dis))
        num_count[idx[0].item()] += 1

    ###############################################################################################
    #                                              可视化                                      #####
    ###############################################################################################
    pose = np.eye(4)
    pose[:3, :3] = my_r  # 旋转矩阵 (3x3)
    pose[:3, 3] = my_t.squeeze()  # 平移向量 (3,)


    np_img = ori_img[0].cpu().numpy() # 图像准备
    print("Original shape:", np_img.shape)
    # np_img = np.transpose(np_img, (2,0,1))  # 变成 (H, W, 3)
    np_img = 255 - np_img # 图像反转（正片恢复）
    np_img = (np_img * 255).astype(np.uint8) # numpy转为0-255

    # ------------------------- #
    # 可视化坐标轴和包围框绘制部分
    # ------------------------- #
    img_draw = np_img.copy() # 复制一份图像用于画图（避免原图受修改）
    # 可视化1：绘制坐标系
    draw_coordinate_axis(img_draw, pose, camera_matrix,axis_length, axis_colors) # 画出坐标系
    # 可视化2：绘制3D包围框
    bbox_corners = get_3d_obb_corners(model_points) # 获取gt模型的3D Box
    draw_3d_bbox(img_draw, bbox_corners, pose, camera_matrix,color=bbox_color, thickness=1) # 画出包围盒

    bbox_corners2 = get_3d_obb_corners(target) # 获取gt模型的3D Box
    draw_3d_bbox(img_draw, bbox_corners2, np.eye(4), camera_matrix,color=bbox_color2, thickness=1) # 画出包围盒
    print("包围盒角点坐标 (单位: 米):\n", bbox_corners)
    print("包围盒尺寸 (X, Y, Z):",
          np.ptp(bbox_corners[:, 0]),  # X轴长度
          np.ptp(bbox_corners[:, 1]),  # Y轴长度
          np.ptp(bbox_corners[:, 2]))  # Z轴长度
    # ------------------------- #
    # 用 Matplotlib 显示最终图像
    # ------------------------- #
    from datetime import datetime
    plt.figure(figsize=(8, 6))
    save_path = os.path.join(save_dir, f"pose_visualization_{frame_id}.png")

    plt.imsave(save_path, img_draw)
    print(f"图像已保存至: {save_path}")
    plt.close()
    #########################################################


end_time = time.time()
total_time = end_time - start_time

fps = total_frames / total_time
print(f"端到端 FPS: {fps:.2f}")

for i in range(num_objects):
    print('Object {0} success rate: {1}'.format(objlist[i], float(success_count[i]) / num_count[i]))
    fw.write('Object {0} success rate: {1}\n'.format(objlist[i], float(success_count[i]) / num_count[i]))
print('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
fw.close()
