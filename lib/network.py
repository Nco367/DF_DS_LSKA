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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.bn4 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print("输入:", x.shape)

        identity = self.shortcut(x)
        print("shortcut(identity):", identity.shape)

        out = self.conv1(x)
        print("conv1:", out.shape)

        out = self.bn1(out)
        print("bn1:", out.shape)

        out = out + self.bn2(identity)
        print("bn1 + bn2(identity):", out.shape)

        out = self.relu(out)
        print("ReLU1:", out.shape)

        identity2 = out
        print("identity2:", identity2.shape)

        out = self.conv2(out)
        print("conv2:", out.shape)

        out = self.bn3(out)
        print("bn3:", out.shape)

        out = out + self.bn4(identity2)
        print("bn3 + bn4(identity2):", out.shape)

        out = identity2 + out
        print("残差连接(identity2 + out):", out.shape)

        out = self.relu(out)
        print("ReLU2:", out.shape)

        return out



class ResPointNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super(ResPointNet, self).__init__()
        # 两个 Rep-Res
        self.rep1 = ResBlock(in_channels, 64)
        self.rep2 = ResBlock(64, 128)

        # Conv + BN + ReLU
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # Conv + BN
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim)
        )

        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, in_channels, N)
        out = self.rep1(x)
        out = self.rep2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        ap_out = self.pool(out)  # (B, feature_dim, 1)
        out = ap_out + out
        out = out.squeeze(-1)  # 输出特征: torch.Size([1, 256, 1024])
        return out


# 测试
if __name__ == "__main__":
    x = torch.randn(8, 3, 1024)  # (batch, xyz, 点数)
    model = ResPointNet(in_channels=3, feature_dim=256)
    y = model(x)
    print("输入:", x.shape)
    print("输出特征:", y.shape)




class CrossAttention(nn.Module):
    def __init__(self, dim_pc, dim_img):
        super().__init__()
        self.q_pc = nn.Linear(dim_pc, dim_pc)
        self.k_img = nn.Linear(dim_img, dim_pc)
        self.v_img = nn.Linear(dim_img, dim_pc)
        self.scale = dim_pc ** 0.5

    def forward(self, feat_pc, feat_img):
        # feat_pc: (B, N, C_pc), feat_img: (B, N, C_img)
        # 支持输入 (B, C, N)，自动转成 (B, N, C)
        # 如果输入是 (B, C, N)，转成 (B, N, C)
        feat_pc = feat_pc.transpose(2,1).contiguous()
        feat_img = feat_img.transpose(2, 1).contiguous()
        Q = self.q_pc(feat_pc)          # (B, N, C_pc)
        K = self.k_img(feat_img)        # (B, N, C_pc)
        V = self.v_img(feat_img)        # (B, N, C_pc)

        attn = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)      # (B, N, C_pc)
        return feat_pc + out              # 残差连接

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        # 这里实例化 CrossAttention
        self.cross1 = CrossAttention(dim_pc=64, dim_img=64)
        self.cross2 = CrossAttention(dim_pc=128, dim_img=128)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        # print("输入 x:", x.shape)  # (B, 3, N) 初始点云特征
        # print("输入 emb:", emb.shape)  # (B, 3, N) 或其他特征

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1) # 64 + 64 ([1, 128, 700])

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x , emb ], dim=1)  # 128 + 128 ([1, 256, 700])

        x = F.relu(self.conv5(pointfeat_2))

        x = F.relu(self.conv6(x))
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        # print("ap_x", ap_x.shape)
        fused_feat = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)

        return fused_feat #128 + 256 + 1024  # 64 128 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj

        self.cnn = ModifiedResnet()
        self.cld_feat = ResPointNet()
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
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        out_img = self.cnn(img) # 特征提取器
        bs, di, _, _ = out_img.size() # 获取输出张量 out_img 的尺寸
        emb = out_img.view(bs, di, -1)

        choose = choose.repeat(1, di, 1) # 是一个选择索引张量, 每个通道都会选择相同的索引
        emb = torch.gather(emb, 2, choose).contiguous() # 根据 choose 中的索引选取元素, 得到每个特征图上特定位置的值

        x = x.transpose(1, 2).contiguous()  # (B, 3, N)
        # x = self.pointnet2(x.transpose(1, 2).contiguous())  # 传入 PointNet2MSG 期望 (B, N, 3 + input_channels)
        # # 如果返回是四维，可以 squeeze 掉最后一维：
        # if x.dim() == 4:
        #     x = x.squeeze(-1)
        # x = self.cld_feat(x)

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
        self.conv1 = torch.nn.Conv2d(6, 64, 1)
        # self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)


        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1408, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)
        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))
        # --- 全局池化 ---
        ap_x = self.ap1(x)
        # --- 注入全局特征到每个点 ---
        ap_x = ap_x.repeat(1, 1, self.num_points)  # (B, 1024, N)
        # --- 拼接所有特征 → (B, 1408, N) --- 拼接所有特征：128 + 256 + 1024 = 1408 ---
        ap_x = torch.cat([pointfeat_1, pointfeat_2, ap_x], dim=1)  # 128+256+1024=1408
        ap_x = ap_x.mean(dim=2)  # (B, 1408)
        return ap_x


class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1408, 512)
        self.conv1_t = torch.nn.Linear(1408, 512)

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

#     # 测试模块
# if __name__ == "__main__":
#     x = torch.randn(8, 64, 1024)  # batch=8, 通道=64, 点数=1024
#     model = RepResBlock(64, 128)
#     y = model(x)
#     print("输入维度:", x.shape)
#     print("输出维度:", y.shape)