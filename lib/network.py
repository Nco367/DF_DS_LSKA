import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet


############################################################################


############################################################################
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}
# backend——主干网络是 ResNet-18。
# sizes——使用金字塔池化模块对多尺度特征进行提取，池化尺寸为 (1, 2, 3, 6)。
# psp_size——金字塔模块的输出特征大小为 512。
# deep_features_size——深度特征分支的输出特征大小为 256。

# ModifiedResnet 负责提取原始数据中的高级特征，通常是多维的特征表示（例如，点云的高级表示、图像的深层次特征等）。


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn, radius

class ModifiedResnet(nn.Module): # 一个使用 ResNet18 架构的神经网络
    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PointTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)

    def forward(self, x):
        # Assume pointcloud_features: (batch_size, num_points, input_dim)
        # Assume image_features: (batch_size, num_points, input_dim)——(seq_len, batch_size, embedding_dim)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        attention_out, _ = self.attention(x, x, x)
        attention_out = attention_out.transpose(0, 1)
        attention_out = attention_out.transpose(1, 2)  # (batch_size, num_points, input_dim)

        return attention_out


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()

        # self.k = 10
        # self.conv1 = torch.nn.Conv2d(6, 64, kernel_size=1, bias=False)
        # self.conv2 = torch.nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        # self.att_layer1 = PointTransformer(128,128) # ————————————————————————————————添加
        # self.att_layer2 = PointTransformer(256, 256) # ————————————————————————————————添加

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

        # x 是从输入的 3 维点云数据中提取特征
        # emb 在这里是从图像中提取的特征
        # conv5 conv6 这两层是在前面的卷积层的基础上进一步处理融合后的特征。它们的作用是逐渐增加特征维度，生成更高级的特征表示
        # ap1：这是一个全局平均池化层

    def forward(self, x, emb):
        # x = get_graph_feature(x, k=self.k)  # 图卷积的动态生成
        # x = self.conv1(x)  # 每一层图卷积处理后的特征
        # x1 = x.max(dim=-1, keepdim=False)[0]  # 都通过最大池化（max(dim=-1)）进行降维，以获取每个点的全局特征

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x , emb), dim=1)

        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv2(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x , emb), dim=1)

        # Step 2: 应用自注意力机制融合点云特征和图像特征
        # pointfeat_1_att = self.att_layer1(pointfeat_1)  # 转置以匹配 MultiheadAttention 的输入要求 # ————————————————————————————————添加
        # pointfeat_2_att = self.att_layer2(pointfeat_2)  # ————————————————————————————————     # ————————————————————————————————添加

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import knn_graph, EdgeConv  # 需要安装torch_geometric
#
#
# class PoseNetFeat(nn.Module):
#     def __init__(self, num_points, k=20):
#         super(PoseNetFeat, self).__init__()
#         self.k = k  # 定义KNN邻域点数
#
#         # 点云特征提取分支（使用图卷积）
#         self.g_conv1 = EdgeConv(nn.Sequential(
#             nn.Linear(6, 64),  # 输入特征：坐标差(3) + 原始坐标(3)
#             nn.ReLU(),
#             nn.Linear(64, 64)
#         ), 'max')
#
#         self.g_conv2 = EdgeConv(nn.Sequential(
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128)
#         ), 'max')
#
#         # 图像特征处理分支（保持原结构）
#         self.e_conv1 = nn.Conv1d(32, 64, 1)
#         self.e_conv2 = nn.Conv1d(64, 128, 1)
#
#         # 全局特征增强
#         self.conv5 = nn.Conv1d(256, 512, 1)
#         self.conv6 = nn.Conv1d(512, 1024, 1)
#         self.ap1 = nn.AvgPool1d(num_points)
#         self.num_points = num_points
#
#     def get_graph_feature(self, x):
#         # 生成KNN图结构
#         batch_size, _, num_points = x.size()
#         x = x.permute(0, 2, 1)  # [B, N, 3]
#         edge_index = knn_graph(x.reshape(-1, 3), k=self.k,
#                                batch=torch.repeat_interleave(torch.arange(batch_size), num_points))
#         return edge_index
#
#     def forward(self, x, emb):
#         # 点云局部几何特征提取
#         edge_index = self.get_graph_feature(x)
#         x = x.permute(0, 2, 1)  # [B, N, 3]
#         x1 = self.g_conv1(x, edge_index).permute(0, 2, 1)  # [B, 64, N]
#
#         # 图像特征处理
#         emb = F.relu(self.e_conv1(emb))
#         pointfeat_1 = torch.cat([x1, emb], dim=1)
#
#         # 第二层图卷积
#         x2 = self.g_conv2(pointfeat_1.permute(0, 2, 1), edge_index).permute(0, 2, 1)
#         emb = F.relu(self.e_conv2(emb))
#         pointfeat_2 = torch.cat([x2, emb], dim=1)
#
#         # 全局特征聚合
#         x = F.relu(self.conv5(pointfeat_2))
#         x = F.relu(self.conv6(x))
#         ap_x = self.ap1(x).repeat(1, 1, self.num_points)
#
#         return torch.cat([pointfeat_1, pointfeat_2, ap_x], dim=1)

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj

        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

    def forward(self, img, x, choose, obj):

        out_img = self.cnn(img) # 特征提取器
        bs, di, _, _ = out_img.size() # 获取输出张量 out_img 的尺寸
        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1) # 是一个选择索引张量, 每个通道都会选择相同的索引
        # emb 是RGB的特征
        emb = torch.gather(emb, 2, choose).contiguous() # 根据 choose 中的索引选取元素, 得到每个特征图上特定位置的值

        x = x.transpose(2, 1).contiguous()  # 维度转置操作
        ap_x = self.feat(x, emb) # 融合

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach()

class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        # self.att_layer1 = PointTransformer(128,128)
        # self.att_layer2 = PointTransformer(256, 256)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x)) # ————————————————————————————————————————————————————————————————改
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x)) # ————————————————————————————————————————————————————————————————改
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
