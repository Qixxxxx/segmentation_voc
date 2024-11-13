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


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes))


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


stages_suffixes = {0: '_conv',
                   1: '_conv_relu_varout_dimred'}


class RCUBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x


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
        self.do = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.final_seg1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.final_seg1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        l1, l2, l3, l4 = self.backbone(x)   # backbone输出
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)

        seg_final = self.final_seg1(x1)
        output = F.interpolate(seg_final, size=input_size, mode='bilinear', align_corners=True)

        # 使用辅助loss
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x3)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

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