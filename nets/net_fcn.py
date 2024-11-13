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
        x = self.layer2(x_1)
        aux = self.layer3(x)
        x = self.layer4(aux)
        return x_1, aux, x


class Resnet101(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet101, self).__init__()
        from functools import partial
        model = resnet101(pretrained)
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
        x = self.layer2(x_1)
        aux = self.layer3(x)
        x = self.layer4(aux)
        return x_1, aux, x

# 网络结构
class MyNet(nn.Module):
    def __init__(self, num_classes, down_tate, backbone='resnet50', pretrained=True, aux_branch=True):
        '''num_classes: 分类数, down_tate： 经过特征提取后的下采样倍数, backbone：特征提取网络的使用, pretrained：是否加载预训练权重,
         aux_branch：是否使用辅助loss'''
        super(MyNet, self).__init__()
        # 根据特征提取网络的输出维度确定psp_module的输入维度
        if backbone == "resnet50":
            self.backbone = Resnet50(down_tate, pretrained)
            m_1_channel = 256
            out_channel = 2048
        elif backbone == "resnet101":
            self.backbone = Resnet101(down_tate, pretrained)
            m_1_channel = 256
            out_channel = 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet50, resnet101.'.format(backbone))

        # for p in self.parameters():
        #     p.requires_grad = False

        self.m1_down = nn.Conv2d(256, 64, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(2048+64, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1, bias=False))

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.final)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        m1, aux, backbone_out = self.backbone(x)   # backbone输出
        fine_size = (m1.size()[2], m1.size()[3])

        m1 = self.m1_down(m1)
        fusion = torch.cat([F.interpolate(backbone_out, fine_size, mode='bilinear', align_corners=True), m1], dim=1)
        seg_final = self.final(fusion)
        output = F.interpolate(seg_final, size=input_size, mode='bilinear', align_corners=True)

        # 使用辅助loss
        if self.aux_branch:
            output_aux = self.auxiliary_branch(aux)
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