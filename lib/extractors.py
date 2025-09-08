from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=bias)

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

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        # self.se1 = SEBlock(inplanes)# 通道注意力
        #self.cbam1 = CBAMBlock(inplanes)# SEBlock 的表现不如预期，且计算资源允许，再换这个；依赖空间信息

        # self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.conv1 = DepthwiseSeparableConv(in_channels=inplanes, out_channels=planes, # ————————————————————————————————添加
                        kernel_size=3, stride=stride, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.conv2 = DepthwiseSeparableConv(in_channels=planes, out_channels=planes, # ————————————————————————————————添加
                      kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = downsample
        self.stride = stride

        # self.se2= SEBlock(planes)
        # self.cbam2 = CBAMBlock(planes)


    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        # x=self.se1(x)
        #x=self.cbam1(x)

        out = self.conv1(x)
        # out= self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out= self.bn2(out)
        # out=self.se2(out)
        #out=self.cbam2(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3

def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def resnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model
