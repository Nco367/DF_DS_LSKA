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
    def __init__(self, in_channels, out_channels,num_groups=8):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.gn1(out)
        out = self.conv2(out)
        out = self.gn1(out)
        out += identity  # 只加一次残差
        out = self.relu(out)
        # print("ResBlock out mean:", out.mean().item(), "std:", out.std().item())
        return out
# 如果想做更多层次残差，应该堆多个 ResBlock，而不是在一个 block 里重复加


class ResPointNet(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super(ResPointNet, self).__init__()
        # 两个 Rep-Res
        self.rep1 = ResBlock(in_channels, 64)
        self.rep2 = ResBlock(64, 128)

        # Conv + BN + ReLU
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True)
        )

        # Conv + BN
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, feature_dim, kernel_size=1, bias=False),
        )

        # 全局池化
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, in_channels, N)

        out_64 = self.rep1(x)
        out_128 = self.rep2(out_64)
        out_128 = self.conv1(out_128)
        # out = self.conv2(out)
        # out = self.pool(out)  # (B, feature_dim, 1)
        # out = ap_out.repeat(1, 1, out.size(-1))  # 重复到每个点  全局池化特征 → 不推荐直接加，推荐重复拼接##########
        # out = torch.cat([out, ap_out], dim=1)  # 拼接而不是相加
        # out = ap_out + out
        #out = out.squeeze(-1)  # 输出特征: torch.Size([1, 256, 1024])

        return out_64,out_128



# 加了ResPointNet
class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.e_conv3 = torch.nn.Conv1d(128, 256, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)

        self.num_points = num_points

    def forward(self, out_64,out_128, emb):
        # print("输入 ori_x:", ori_x.shape)  # (B, 3, N) 初始点云特征
        # print("输入 x_128:", x_128.shape)  # (B, 3, N) 或其他特征

        out_64 = out_64
        emb_64 = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([out_64, emb_64], dim=1) # 64 64

        out_128 = out_128 # 128
        emb_128 = F.relu(self.e_conv2(emb_64))
        pointfeat_2 = torch.cat([out_128 , emb_128 ], dim=1)  # 128 128

        x = F.relu(self.conv5(pointfeat_2)) # 1024
        x = F.relu(self.conv6(x)) # 1024

        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        fused_feat = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) # 128 + 256 + 1024

        return fused_feat




# 加了ResPointNet
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

        # # ===== 局部分支（轻量） =====
        # self.local_r = nn.Sequential(
        #     nn.Conv1d(512, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, num_obj*4, 1)
        # )
        # self.local_t = nn.Sequential(
        #     nn.Conv1d(512, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, num_obj*3, 1)
        # )
        # self.local_c = nn.Sequential(
        #     nn.Conv1d(512, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, num_obj*1, 1)
        # )



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


        out_64,out_128= self.cld_feat(x)
        ap_x = self.feat(out_64,out_128, emb) # 融合
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

        # # ===== 局部分支 =====
        # pooled_local = F.adaptive_avg_pool1d(pointfeat_2, self.num_points)  # (B, 512, N)
        # rx_l = self.local_r(pooled_local).view(bs, self.num_obj, 4, self.num_points)
        # tx_l = self.local_t(pooled_local).view(bs, self.num_obj, 3, self.num_points)
        # cx_l = torch.sigmoid(self.local_c(pooled_local)).view(bs, self.num_obj, 1, self.num_points)
        #
        # # ===== 融合（全局 + 局部） =====
        # rx = rx_g + rx_l
        # tx = tx_g + tx_l
        # cx = cx_g + cx_l


        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx, emb.detach()



# 加了ResPointNet
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

    def forward(self, out_64,out_128, emb):

        out_64 = out_64
        # x_64 = F.relu(self.conv1(ori_x))
        emb_64 = F.relu(self.e_conv1(emb))
        # x_128_mod = F.relu(self.bn1(self.film1(x_128, emb)))
        pointfeat_1 = torch.cat([out_64, emb_64], dim=1) # 64 64

        out_128 = out_128 # 128
        emb_128 = F.relu(self.e_conv2(emb_64))
        # pointfeat_2 = F.relu(self.bn2(self.film2(x, emb)))
        pointfeat_2 = torch.cat([out_128 , emb_128 ], dim=1)  # 128 128
        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1) # 64+128

        x = F.relu(self.conv5(pointfeat_3))# 512
        x = F.relu(self.conv6(x))
        # --- 全局池化 ---
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024)
        return ap_x



class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.cld_feat = ResPointNet()
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

        out_64,out_128= self.cld_feat(x)

        ap_x = self.feat(out_64,out_128, emb) # 融合



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