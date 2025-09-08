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

class Pointnet2MSG(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(
        self, input_channels=0, use_xyz=True
    ):
        super(Pointnet2MSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=700,
                radii=[0.0175, 0.025],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256 + 256

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + input_channels, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq

        _, N, _ = pointcloud.size()
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]

class CrossAttention(nn.Module):
    def __init__(self, dim_pc, dim_img):
        super().__init__()
        self.q_pc = nn.Linear(dim_pc, dim_pc)
        self.k_img = nn.Linear(dim_img, dim_pc)
        self.v_img = nn.Linear(dim_img, dim_pc)
        self.scale = dim_pc ** 0.5

    def forward(self, feat_pc, feat_img):
        # feat_pc: (B, N, C_pc), feat_img: (B, N, C_img)
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
        self.conv2 = torch.nn.Conv1d(128, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):

        # x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x , emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x , emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        pointfeat_1 = CrossAttention(pointfeat_1, emb)
        pointfeat_2 = CrossAttention(pointfeat_2, emb)
        fused_feat = torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)

        return fused_feat #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj

        self.cnn = ModifiedResnet()
        self.pointnet2 = Pointnet2MSG(input_channels=0,use_xyz=True)
        self.feat = PoseNetFeat(num_points)

        self.conv1_r = torch.nn.Conv1d(1472, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1472, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1472, 640, 1)

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


        x = x.transpose(2, 1).contiguous()  # (B, 3, N)
        x = self.pointnet2(x.transpose(1, 2).contiguous())  # 传入 PointNet2MSG 期望 (B, N, 3 + input_channels)
        # 如果返回是四维，可以 squeeze 掉最后一维：
        if x.dim() == 4:
            x = x.squeeze(-1)
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
if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    import torch.optim as optim

    B = 2
    N = 32
    inputs = torch.randn(B, N, 6).cuda()
    labels = torch.from_numpy(np.random.randint(0, 3, size=B * N)).view(B, N).cuda()
    model = Pointnet2MSG(3)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Testing with xyz")