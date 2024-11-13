#-------------------------------------#
#       对图片进行预测

from nets.net_unetpp import MyNet as mynet
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import colorsys
import torch
import copy
import os
import warnings
warnings.filterwarnings(action='ignore')

class MyNet(object):
    # -----------------------------------------#
    #   注意修改model_path、num_classes
    #   和backbone
    #   使其符合自己的模型
    # -----------------------------------------#
    _defaults = {
        "model_path":       'model_data/unetpp_75.12.pth',
        "model_image_size": (512, 512, 3),
        "backbone":          "resnet50",
        "downsample_factor": 32,
        "num_classes":       21,
        "cuda":             True,
        "blend":            False,
    }

    # ---------------------------------------------------#
    #   初始化NET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.net = mynet(num_classes=self.num_classes, down_tate=self.downsample_factor,
                         pretrained=False, backbone=self.backbone, aux_branch=False)
        self.net = self.net.eval()

        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict, strict=False)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128)]
        else:
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size  # 原始图像的尺寸
        w, h = size  # 目标图像的尺寸
        scale = min(w / iw, h / ih)  # 转换的最小比例
        # 保证长或宽至少一个符合目标图像尺寸
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)  # 三次样条插值
        new_image = Image.new('RGB', size, (128, 128, 128))  # 创建一个size灰色图像（128，128，128）灰色图像颜色值
        # // 为整数除法，计算图像的位置
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 把插值后的图片粘贴到灰色图中
        return new_image, nw, nh

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        old_img = copy.deepcopy(image)  # 深拷贝，备份
        orininal_h = np.array(image).shape[0]  # 图像的高
        orininal_w = np.array(image).shape[1]  # 图像的宽

        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))  # 不失真的resize
        images = [np.array(image) / 255]  # 归一化
        images = np.transpose(images, (0, 3, 1, 2))

        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                 int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        if self.blend:
            image = Image.blend(old_img, image, 0.3)  # 原图像和分割图像混合

        return image

pspnet = MyNet()

imgs = os.listdir("./test_jpg")
for jpg in imgs:
    img = Image.open(("./test_jpg/" + jpg))
    r_image = pspnet.detect_image(img)
    r_image.save("./test_out/" + jpg)
    print(" done!")