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
from lib.pointnet2_utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

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
# from torch_geometric.nn import knn, radius

class ModifiedResnet(nn.Module): # 一个使用 ResNet18 架构的神经网络
    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

# DS-LSKA
class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self,x, emb):
        x = F.relu(self.conv1(x))
        emb_64 = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb_64], dim=1) # 64 64

        x = F.relu(self.conv2(x))
        emb_128 = F.relu(self.e_conv2(emb_64))
        pointfeat_2 = torch.cat([x , emb_128 ], dim=1)  # 128 128

        x = F.relu(self.conv5(pointfeat_2)) # 1024
        x = F.relu(self.conv6(x)) # 1024

        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        fused_feat = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) # 128 + 256 + 1024
        return fused_feat


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

        self.conv4_r = torch.nn.Conv1d(128, num_obj * 4, 1)  # quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj * 3, 1)  # translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj * 1, 1)  # confidence

    def forward(self, img, x, choose, obj):

        out_img = self.cnn(img)  # 特征提取器
        bs, di, _, _ = out_img.size()  # 获取输出张量 out_img 的尺寸
        emb = out_img.view(bs, di, -1)

        choose = choose.repeat(1, di, 1)  # 是一个选择索引张量, 每个通道都会选择相同的索引
        emb = torch.gather(emb, 2, choose).contiguous()  # 根据 choose 中的索引选取元素, 得到每个特征图上特定位置的值

        x = x.transpose(2, 1).contiguous()

        ap_x = self.feat(x, emb)  # 融合
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

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb_64 = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb_64], dim=1)  # 64 64

        x = F.relu(self.conv2(x))
        emb_128 = F.relu(self.e_conv2(emb_64))
        pointfeat_2 = torch.cat([x, emb_128], dim=1)  # 128 128
        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)  # 128+256=384

        x = F.relu(self.conv5(pointfeat_3))  # 384
        x = F.relu(self.conv6(x))
        # --- 全局池化 ---
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

        self.conv3_r = torch.nn.Linear(128, num_obj * 4)  # quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj * 3)  # translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]

        x = x.transpose(2, 1).contiguous()

        ap_x = self.feat(x, emb)  # 融合

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

#     # 测试模块
# if __name__ == "__main__":
#     x = torch.randn(8, 64, 1024)  # batch=8, 通道=64, 点数=1024
#     model = RepResBlock(64, 128)
#     y = model(x)
#     print("输入维度:", x.shape)
#     print("输出维度:", y.shape)