import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
import matplotlib.pyplot as plt

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]# 对象列表～________________________________________________________________________________________________________________________________________________________________
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = [] #
        self.meta = {} #
        self.pt = {} # points
        self.root = root #
        self.noise_trans = noise_trans #
        self.refine = refine #

        # 根据数据集添加了1图像 2深度图 3掩码
        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                # 物体ID 累加
                item_count += 1

                # 加载训练帧、测试帧编号列表
                input_line = input_file.readline()
                print(f"读取行: {input_line.strip()}")  # 添加调试输出
                # 检查物体数量是否>10
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                # 检查最后一个元素是否为换行符
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                # 添加RGB、DEPTH图像地址至列表
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 如果是评估模式，添加掩码图像地址至列表
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                # 添加物体ID 到list_obj、添加物体的帧编号到list_rank
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            # 加载模型的ground truth文件
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            meta = yaml.load(meta_file, Loader=yaml.FullLoader)
            self.meta[item] = {int(k): v for k, v in meta.items()} # 将字典的犍（帧编号）改为整数，而不是字符串
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))

            # print("Loaded meta for obj", item)
            # print("Sample keys:", self.meta[item].keys())
            # print("Object {0} buffer loaded".format(item))
            # print("物体 {0} 的点云形状: {1}".format(item, self.pt[item].shape))

        self.length = len(self.list_rgb) # 视频帧长度

        # 相机内参
        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        # 创建地图
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])


        self.num = num #
        self.add_noise = add_noise #
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) #调用 transform(image) 时，图像的亮度、对比度、饱和度和色调都会被随机调整
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 会进一步处理图像，调整每个通道的均值和标准差，使其符合标准正态分布
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx =[7, 8]


    def __getitem__(self, index):

        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]
        # print("Current obj:", obj)
        # print("Current rank:", rank)
        # print("Available ranks in meta for obj:", self.meta[obj].keys())
        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # 掩码不为0的点
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255))) # 掩码为255的点
        else:
            # mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))   # 自制数据集：开启
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0] # 掩码所有红色通道为255的像素
        mask = mask_label * mask_depth # 两个掩码的交集

        # ——————————————————————————————————————————————————img_masked——————————————————————————————————————————————————————
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img
        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb']) # 自制数据集：关闭
            # rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label)) # 自制数据集：开启
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        # p_img = np.transpose(img_masked, (1, 2, 0))
        # plt.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # ——————————————————————————————————————————————————choose——————————————————————————————————————————————————————
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # 数据集中过来的掩码，创建数组 ；[0] 索引表示 提取元组中的第一个数组
        if len(choose) == 0:
            cc = torch.LongTensor([0]) # LongTensor 经常用于存储整数类型的标签或索引
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()] # 每次训练时，输入的点是随机选择的，从而增加了数据的多样性
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # ——————————————————————————————————————————————————cloud——————————————————————————————————————————————————————
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale # 深度值除以相机的深度标定因子
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx # 计算3D点云的X坐标
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy # 计算3D点云的Y坐标
        cloud = np.concatenate((pt0, pt1, pt2), axis=1) # 合并X, Y, Z坐标
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t) # 添加噪声
        #
        # fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # ——————————————————————————————————————————————————model_points——————————————————————————————————————————————————————
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)
        #
        # fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # ——————————————————————————————————————————————————target——————————————————————————————————————————————————————
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0
        #
        # fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


from plyfile import PlyData
import numpy as np

def ply_vtx(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex']
    xyz = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    return np.array(xyz, dtype=np.float32)


# import struct
#
# #
# def ply_vtx(path):
#     with open(path, 'rb') as f:  # 注意以二进制模式打开
#         header = []
#         while True:
#             line = f.readline().decode('utf-8').strip()
#             header.append(line)
#             if line == 'end_header':
#                 break
#
#         # 解析顶点数量
#         vertex_line = [line for line in header if line.startswith('element vertex')][0]
#         vertex_count = int(vertex_line.split()[-1])
#
#         # 计算每个顶点的字节长度（仅读取x, y, z）
#         # 假设顶点属性顺序为 x, y, z（后续属性跳过）
#         vertex_format = 'fff'  # 3个float（x, y, z）
#         vertex_size = struct.calcsize(vertex_format)
#
#         # 读取顶点数据
#         pts = []
#         for _ in range(vertex_count):
#             data = f.read(vertex_size)
#             x, y, z = struct.unpack(vertex_format, data)
#             pts.append([x, y, z])
#             # 跳过后续属性（根据实际文件调整）
#             # 如果你的文件每个顶点还有其他属性（如颜色），需计算总长度
#             # 例如：若每个顶点总长度为 3*float + 4*uchar，则总长度=12+4=16
#             # 使用 f.read(总长度 - vertex_size) 跳过后续字节
#             # f.read(16 - 12)  # 跳过4字节
#
#         return np.array(pts, dtype=np.float32)
