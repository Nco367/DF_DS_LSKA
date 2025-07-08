import torch
from torch import nn
import torchvision.models as models
from torch.nn import functional as F
import lib.extractors as extractors

##########################################################################
class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x
class CustomResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(CustomResNet18, self).__init__()

        # 载入预训练的 ResNet-18
        resnet18 = models.resnet18(pretrained=pretrained)

        # 修改第一个卷积层（例如，改变卷积核的大小或通道数）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)

        # 其他层可以直接使用原来的结构
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # 如果需要修改全连接层
        self.fc = resnet18.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.W = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        theta_x = self.theta(x).view(batch_size, -1, H * W)
        phi_x = self.phi(x).view(batch_size, -1, H * W)
        g_x = self.g(x).view(batch_size, -1, H * W)
        theta_phi = torch.bmm(theta_x.permute(0, 2, 1), phi_x)
        theta_phi = torch.softmax(theta_phi, dim=-1)
        out = torch.bmm(g_x, theta_phi)
        out = out.view(batch_size, C // 2, H, W)
        out = self.W(out)
        return out + x
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
#         super(BasicBlock, self).__init__()
#         self.conv1 = DepthwiseSeparableConv(in_channels=in_channel, out_channels=out_channel,
#                       kernel_size=3, stride=stride, padding=1, bias=False)
#         # self.conv1 = nn.Sequential(
#         #     InceptionBlock(in_channel=in_channel, out_channel=in_channel),
#         #     nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#         #               kernel_size=3, stride=stride, padding=1, bias=False),
#         # )
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.relu = nn.ReLU()
#         self.conv2 = DepthwiseSeparableConv(in_channels=out_channel, out_channels=out_channel,
#                       kernel_size=3, stride=1, padding=1, bias=False)
#         # self.conv2 = nn.Sequential(
#         #     InceptionBlock(in_channel=out_channel, out_channel=out_channel),
#         #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#         #               kernel_size=3, stride=1, padding=1, bias=False)
#         # )
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#         self.se1 = NonLocalBlock(in_channel)
#         self.se2 = NonLocalBlock(out_channel)
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         x = self.se1(x)
#
#         out = self.conv1(x)
#         # out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         # print(out.shape)
#         # out = self.bn2(out)
#
#         out = self.se2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
##########################################################################

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 卷积层
            nn.PReLU()  # 激活函数
        )

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # 卷积和激活
        x_up = self.conv(x_up)
        return x_up

class LSKA(nn.Module):
    def __init__(self, dim, k_size):
        super(LSKA,self).__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)
        #  dim：表示输入和输出的通道数
        #  kernel_size：卷积核的大小，通常是一个整数或一个整数元组。
        #  stride：卷积核的步幅
        #  padding：填充，指在输入特征图的边缘添加的额外像素数，以控制输出尺寸。
        #  groups：分组卷积的数量。
        #  dilation：膨胀卷积的膨胀系数。

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=False):
        super(PSPNet, self).__init__()
        self.feats = getattr(extractors, backend)(pretrained) # resnet**  动态加载一个主干网络
        # self.lska = LSKA(psp_size,7) # ——————1 主干网络后添加LSKA、使用 LSKA 模块，处理主干提取的特征图
        self.psp = PSPModule(psp_size, 1024, sizes) # 调用金字塔池化模块（PSPModule）进行多尺度上下文融合
        # self.lska = LSKA(1024,53) # ——————2 PSP后添加LSKA  #_______————————————————————————————————————————————————————添加
        self.drop_1 = nn.Dropout2d(p=0.3) # 应用 Dropout 减少过拟合风险

        self.up_1 = PSPUpsample(1024, 256)
        # self.lska_1 = LSKA(256,53)
        self.up_2 = PSPUpsample(256, 64)
        # self.lska_2 = LSKA(64,53)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # self.lska_final = LSKA(64,53) # #_______————————————————————————————————————————————————————添加

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)  # 调用主干特征提取器
        # f = self.lska(f) # ——————1 对主干网络提取的特征图 f 进行局部自适应特征增强
        p = self.psp(f) # 调用 Pyramid Pooling 模块（PSPModule）进行多尺度特征融合
        # p = self.lska(p) #——————2 #_______————————————————————————————————————————————————————添加
        p = self.drop_1(p) # 用于在训练阶段随机丢弃输入特征图中的某些激活值

        p = self.up_1(p)
        # p = self.lska_1(p) # ————3 上采样1
        p = self.drop_2(p)

        p = self.up_2(p)
        # p = self.lska_2(p) # ————3 上采样2
        p = self.drop_2(p)

        p = self.up_3(p)
        # p = self.lska_final(p)# ————4 最终卷积层 #_______————————————————————————————————————————————————————添加
        
        return self.final(p)