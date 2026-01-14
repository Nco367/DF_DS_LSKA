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
from datasets.linemod6_eval.dataset import PoseDataset as PoseDataset_linemod6
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

# ====================== 关键修改1：统一尺度因子 ======================
# 定义尺度转换因子（根据Linemod数据集特性，原始点云通常是毫米，转换为米）
SCALE_FACTOR = 1.0 / 1000.0  # 毫米 -> 米
# 或者根据你的diameter缩放逻辑调整：SCALE_FACTOR = 0.1 / 1000.0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='/media/q/SSD2T/1linux/Linemod6_dataset/Linemod_preprocessed',
                    help='dataset root dir')
parser.add_argument('--model', type=str,
                    default='/media/q/SSD2T/1linux/Linemod6_trained_mod/pose_model_2_0.012589813495681558.pth',
                    help='resume PoseNet model')
parser.add_argument('--refine_model', type=str,
                    default='/media/q/SSD2T/1linux/Linemod6_trained_mod/pose_refine_model_51_0.005152655821685291.pth',
                    help='resume PoseRefineNet model')
opt = parser.parse_args()

# 相机内参（保持不变）
cx = 962.849487
cy = 537.564453
fx = 1267.569092
fy = 1267.200073
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# 可视化参数（保持不变）
vis_size = (1920, 1080)
axis_length = 0.1  # 米（坐标系轴长度）
bbox_color = (0, 255, 122)  # 预测框（绿）
bbox_color2 = (255, 0, 0)  # GT框（红）
axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # XYZ轴颜色

# 模型参数（保持不变）
num_objects = 3
objlist = [3]
num_points = 1000
iteration = 2
bs = 1

dataset_config_dir = '/media/q/SSD2T/1linux/Linemod6_dataset/dataset_config'
output_result_dir = '/media/q/SSD2T/1linux/Linemod6_eval'
knn = KNearestNeighbor(1)

# ====================== 模型加载（优化后，增加鲁棒性） ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
estimator = PoseNet(num_points=num_points, num_obj=num_objects).to(device)
refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects).to(device)


# 安全加载模型
def load_model(model, path, name):
    try:
        state_dict = torch.load(path, map_location=device)
        # 处理多GPU训练的权重
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        print(f"{name} 加载成功")
    except Exception as e:
        print(f"{name} 加载失败: {e}")
        exit(1)


load_model(estimator, opt.model, "PoseNet")
load_model(refiner, opt.refine_model, "PoseRefineNet")

# 打印参数量
estimator_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
refiner_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
print(f"Estimator Params: {estimator_params:,}")
print(f"Refiner Params: {refiner_params:,}")

# 数据加载（保持不变）
testdataset = PoseDataset_linemod6('eval', num_points, False, opt.dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

# 直径加载（保持你的缩放逻辑）
diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file, Loader=yaml.FullLoader)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(f"直径（米）: {diameter}")

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

# 结果保存目录（保持不变）
base_dir = "/media/q/SSD2T/1linux/Linemod6_eval/evaluation_result/"
os.makedirs(base_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join(base_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"结果保存目录: {save_dir}")

# 推理计时（保持不变）
total_frames = len(testdataset)
start_time = time.time()
print('Start inference...')

for i, data in enumerate(testdataloader, 0):
    points, choose, img, target, model_points, idx, ori_img = data
    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue

    # 数据移到GPU（保持不变）
    points, choose, img, target, model_points, idx, ori_img = points.to(device), \
        choose.to(device), \
        img.to(device), \
        target.to(device), \
        model_points.to(device), \
        idx.to(device), \
        ori_img.to(device)

    # 反查原始RGB路径（保持不变）
    rgb_path = testdataset.list_rgb[i]
    frame_id = os.path.splitext(os.path.basename(rgb_path))[0]
    print(f"\n当前样本原始帧号: {frame_id}")

    torch.cuda.empty_cache()
    with torch.no_grad():
        # PoseNet推理（保持不变）
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        # Refiner迭代细化（保持不变）
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
        for ite in range(0, iteration):
            T = torch.from_numpy(my_t.astype(np.float32)).to(device).view(1, 3).repeat(num_points, 1).contiguous().view(
                1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = torch.from_numpy(my_mat[:3, :3].astype(np.float32)).to(device).view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_points = torch.bmm((points - T), R).contiguous()
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

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        # 后处理计算距离（保持不变）
        model_points_np = model_points[0].cpu().detach().numpy()
        my_r_mat = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points_np, my_r_mat.T) + my_t

        target_np = target[0].cpu().detach().numpy()

        if idx[0].item() in sym_list:
            pred_tensor = torch.from_numpy(pred.astype(np.float32)).to(device).transpose(1, 0).contiguous()
            target_tensor = torch.from_numpy(target_np.astype(np.float32)).to(device).transpose(1, 0).contiguous()
            inds = knn.forward(knn.k, target_tensor.unsqueeze(0), pred_tensor.unsqueeze(0))
            target_tensor = torch.index_select(target_tensor, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred_tensor.transpose(1, 0) - target_tensor.transpose(1, 0)), dim=1),
                             dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target_np, axis=1))

        # 成功率统计（保持不变）
        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            print(f'No.{i} Pass! Distance: {dis:.6f}')
            fw.write(f'No.{i} Pass! Distance: {dis:.6f}\n')
        else:
            print(f'No.{i} NOT Pass! Distance: {dis:.6f}')
            fw.write(f'No.{i} NOT Pass! Distance: {dis:.6f}\n')
        num_count[idx[0].item()] += 1

    # ====================== 关键修改2：可视化部分（核心修复） ======================
    # 1. 构建位姿矩阵（保持不变）
    pose = np.eye(4)
    pose[:3, :3] = my_r_mat  # 旋转矩阵 (3x3)
    pose[:3, 3] = my_t.squeeze()  # 平移向量 (3,)

    # 2. 图像预处理（修复图像格式错误）
    np_img = ori_img[0].cpu().numpy()
    print(f"原始图像形状: {np_img.shape}")

    # 修复1：处理图像通道顺序和数值范围
    if np_img.shape[0] == 3:  # 如果是 (C, H, W) 格式
        np_img = np.transpose(np_img, (1, 2, 0))  # 转为 (H, W, C)
    # 修复2：正确的数值归一化（避免像素值溢出）
    np_img = (np_img - np.min(np_img)) / (np.max(np_img) - np.min(np_img) + 1e-8)  # 归一化到0-1
    np_img = (np_img * 255).astype(np.uint8)  # 转为0-255
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)  # 适配OpenCV的BGR格式

    # 3. 尺度转换（核心！解决包围框过大问题）
    # 对模型点云进行尺度缩放（毫米→米）
    model_points_scaled = model_points_np * SCALE_FACTOR
    # 对平移向量也进行相同尺度缩放（保持一致性）
    pose_scaled = np.eye(4)
    pose_scaled[:3, :3] = pose[:3, :3]  # 旋转矩阵不变
    pose_scaled[:3, 3] = pose[:3, 3] * SCALE_FACTOR  # 平移向量缩放
    # 对GT target也进行尺度缩放
    target_scaled = target_np * SCALE_FACTOR

    # 4. 绘制可视化内容
    img_draw = np_img.copy()

    # 绘制预测坐标系（使用缩放后的位姿）
    draw_coordinate_axis(
        img_draw,
        pose_scaled,
        camera_matrix,
        axis_length=axis_length,  # 0.1米（和你的设置一致）
        axis_colors=axis_colors
    )

    # 绘制预测3D包围框（使用缩放后的点云）
    bbox_corners = get_3d_obb_corners(model_points_scaled)
    draw_3d_bbox(
        img_draw,
        bbox_corners,
        pose_scaled,
        camera_matrix,
        color=bbox_color,
        thickness=2  # 加粗便于观察
    )

    # 绘制GT 3D包围框（使用缩放后的target）
    bbox_corners2 = get_3d_obb_corners(target_scaled)
    draw_3d_bbox(
        img_draw,
        bbox_corners2,
        np.eye(4),  # GT位姿为单位矩阵（假设已对齐）
        camera_matrix,
        color=bbox_color2,
        thickness=2
    )

    # 打印调试信息（验证尺度是否正确）
    print(f"缩放后包围盒角点坐标 (米):\n{bbox_corners[:5]}")  # 只打印前5个点
    print(
        f"缩放后包围盒尺寸 (米): X={np.ptp(bbox_corners[:, 0]):.3f}, Y={np.ptp(bbox_corners[:, 1]):.3f}, Z={np.ptp(bbox_corners[:, 2]):.3f}")

    # 5. 保存图像（修复Matplotlib保存格式问题）
    save_path = os.path.join(save_dir, f"pose_visualization_{frame_id}.png")
    # 转换回RGB格式保存（Matplotlib默认RGB）
    img_draw_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))  # 调整画布大小便于观察
    plt.imsave(save_path, img_draw_rgb)
    print(f"图像已保存至: {save_path}")
    plt.close()

# 推理统计（保持不变）
end_time = time.time()
total_time = end_time - start_time
fps = total_frames / total_time
print(f"\n端到端 FPS: {fps:.2f}")
print(f"总推理时间: {total_time:.2f} 秒")

# 成功率输出（保持不变）
for i in range(num_objects):
    if num_count[i] > 0:
        success_rate = float(success_count[i]) / num_count[i]
    else:
        success_rate = 0.0
    print(f'Object {objlist[i]} success rate: {success_rate:.4f} ({success_count[i]}/{num_count[i]})')
    fw.write(f'Object {objlist[i]} success rate: {success_rate:.4f}\n')

total_success = float(sum(success_count)) / sum(num_count) if sum(num_count) > 0 else 0.0
print(f'ALL success rate: {total_success:.4f}')
fw.write(f'ALL success rate: {total_success:.4f}\n')
fw.close()