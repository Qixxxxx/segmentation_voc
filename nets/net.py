import torch
import torch.nn.functional as F
from torch import nn
from nets.resnet import resnet50, resnet101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Resnet50(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet50, self).__init__()
        from functools import partial
        model = resnet50(pretrained)
        # 使用空洞卷积控制最后尺寸
        if dilate_scale == 8:
            # .apply自定义
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))  # partial:对部分参数进行操作
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # 获得预训练好的resnet
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    # 控制使用空洞卷积时的步长
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        return x_1, x_2, x_3, x_4


# ppm模块
class SPPMoudle(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        ''' in_channels: 输入spp模块的通道维度， pool_sizes: 池化核大小组成的list, out_channels:输出通道数'''
        super(SPPMoudle, self).__init__()

        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size)
                                     for pool_size in pool_sizes])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)       # 用全局平均池化将特征图缩放
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 每个stage输出通道一致
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]   # 创建一个list里面已经存在原输入
        # 使用extend将pspmodule产生的特征图添加入列表
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',  # 使用双线性插值插值到输入大小一致
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


# 像素配准模块
class PixelRegistration_l(nn.Module):    # 低分辨率插高分辨率
    def __init__(self, l_inplane, outplane):
        super(PixelRegistration_l, self).__init__()
        self.down_l = nn.Conv2d(l_inplane, outplane, kernel_size=1, bias=False)   # 低分辨率
        self.down_h = nn.Conv2d(outplane, outplane, kernel_size=1, groups=outplane, bias=False)   # 高分辨率
        self.down_f = nn.Conv2d(outplane * 2, outplane, kernel_size=3, padding=1, groups=outplane, bias=False)

        # 生成流场
        self.field_make = nn.Sequential(
            nn.Conv2d(outplane * 2, outplane, kernel_size=1, stride=1, groups=outplane),
            nn.BatchNorm2d(outplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplane, 2, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x1, x2):

        low_feature = x1  # low_featrue为低分辨率图，h_feature为高分辨率图
        h_feature = x2
        h_feature_orign = h_feature
        h, w = h_feature.size()[2:]
        size = (h, w)
        # 对两个层次特征图进行处理
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        # 将低分辨率图进行双线性上采样
        low_feature = F.interpolate(low_feature, size=size, mode="bilinear", align_corners=False)
        # 预测语义流场 === 其实就是输入一个3x3的卷积
        field = self.field_make(torch.cat([h_feature, low_feature], 1))
        # 将Flow Field warp 到当前的高分辨率图
        h_feature = self.warp(h_feature_orign, field, size=size)
        fusion = torch.cat([h_feature, low_feature], 1)
        fusion = self.down_f(fusion)
        return fusion

    def warp(self, inputs, field, size):
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        # 从-1到1等距离生成out_h个点，每一行重复out_w个点，最终生成(out_h, out_w)的像素点
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        # 生成w的转置矩阵
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        # 展开后进行合并
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + field.permute(0, 2, 3, 1) / norm
        # grid指定由input空间维度归一化的采样像素位置，其大部分值应该在[ -1, 1]的范围内
        # 如x=-1,y=-1是input的左上角像素，x=1,y=1是input的右下角像素
        output = F.grid_sample(inputs, grid, align_corners=True)
        return output


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(256, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_orgin = x
        x_d = self.conv1(x)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout, x_d], dim=1)
        x = self.conv2(x)
        sa = self.sigmoid(x)
        x = sa * x_orgin
        return x


# CAM模块
class Cam(nn.Module):
    def __init__(self, k_size=3):
        super(Cam, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# 网络结构
class MyNet(nn.Module):
    def __init__(self, num_classes, down_tate, backbone='resnet50', pretrained=True, aux_branch=True):
        '''num_classes: 分类数, down_tate： 经过特征提取后的下采样倍数, backbone：特征提取网络的使用, pretrained：是否加载预训练权重,
         aux_branch：是否使用辅助loss'''
        super(MyNet, self).__init__()
        # 根据特征提取网络的输出维度确定psp_module的输入维度
        if backbone == "resnet50":
            self.backbone = Resnet50(down_tate, pretrained)
            x_1_channel = 256
            x_2_channel = 512
            x_3_channel = 1024
            fianl_channel = 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet50'.format(backbone))

        self.spp = SPPMoudle(fianl_channel, 512, pool_sizes=[1, 2, 3, 6])

        for p in self.parameters():
            p.requires_grad = False

        self.sa = SpatialAttention(7)
        self.prm_2 = PixelRegistration_l(x_3_channel, x_2_channel)   # 1024,512
        self.prm_3 = PixelRegistration_l(x_2_channel, x_2_channel)   # 512,512
        self.prm_4 = PixelRegistration_l(x_2_channel, x_1_channel)   # 512,256

        # for p in self.parameters():
        #     p.requires_grad = False

        # self.se = SELayer(512+256)
        self.cam = Cam()

        self.down1 = nn.Conv2d(512+256, 256, kernel_size=1, bias=False)
        self.spp_down = nn.Conv2d(512, 256, kernel_size=1, bias=False)

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.final_seg)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x1, x2, x3, x4 = self.backbone(x)   # backbone输出
        fine_size = (x1.size()[2], x1.size()[3])   # 1/4原图大小 256

        spp_out = self.spp(x4)           # 1/16原图大小 512
        branch_1 = self.sa(x1)  # 1/4原图大小  256
        branch_2 = self.prm_2(x3, x2)    # 1/8原图大小  512
        branch_2 = self.prm_3(spp_out, branch_2)   # 1/8原图大小  512
        branch_1 = self.prm_4(branch_2, branch_1)  # 1/4原图大小  256

        spp_out_up = self.spp_down(spp_out)    # 1/8 256
        spp_out_up = F.interpolate(spp_out_up, size=fine_size, mode='bilinear', align_corners=True)  # 1/4 256

        fusion = torch.cat([spp_out_up, branch_1], dim=1)
        fusion = self.cam(fusion)  # 通道注意力

        seg_final = self.final_seg(fusion)
        output = F.interpolate(seg_final, size=input_size, mode='bilinear', align_corners=True)

        # 使用辅助loss
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x3)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()