import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2


# 粗暴的resize会使得图片失真，采用letterbox_image可以较好的解决这个问题。该方法可以保持图片的长宽比例，剩下的部分采用灰色填充
def letterbox_image(image, label, size):   # image：输入的原图, label：标签, size：需要的尺寸
    label = Image.fromarray(np.array(label))   # 将array转换为image
    iw, ih = image.size                        # 原始的宽高
    w, h = size                                # 目标的宽高
    scale = min(w/iw, h/ih)                    # 转换比例
    # 按比例放缩，并将其转换为int
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)   # 使用三线性插值改变图像尺寸
    new_image = Image.new('RGB', size, color=(128, 128, 128))   # 生成一个新RGB三通道的图像，填充灰色
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))        # 按比例将原图粘贴到新图里面，于是最后的目标图就产生了

    label = label.resize((nw, nh), Image.NEAREST)  # 使用最邻近插值改变图像尺寸
    new_label = Image.new('L', size, color=0)          # 生成一个单通道的灰度图，填充黑色
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))

    return new_image, new_label


# rand函数作用：生成一个(a,b)之间随机数
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a     # np.random.rand()返回一个服从（0-1）均匀分布的随机样本值


class NetDataset(Dataset):
    def __init__(self, train_lines, image_size, num_classes, random_data):
        super(NetDataset, self).__init__()

        self.train_lines = train_lines         # 训练样本
        self.train_batches = len(train_lines)  # 训练样本的个数
        self.image_size = image_size           # 输入网络的尺寸
        self.num_classes = num_classes         # 分类数目
        self.random_data = random_data         # 是否数据增强

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 数据随机增强
    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''
        jitter代表原图片的宽高的扭曲比率
        hue=.1，sat=1.5，val=1.5；分别代表hsv色域中三个通道的扭曲，分别是：色调（H），饱和度（S），明度（V）
        '''
        label = Image.fromarray(np.array(label))
        h, w = input_shape              # 输入的高和宽

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)  # 随机生成宽高比

        scale = rand(0.5, 1.5)       # 随机生成缩放比例
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")
        
        # 翻转图像
        flip = rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 将图像多余的部分加上灰条
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), color=(128,128,128))
        new_label = Image.new('L', (w, h), color=0)
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # 颜色抖动  RGB->HVS->RGB
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data, label

    def __getitem__(self, index):
        annotation_line = self.train_lines[index]
        name = annotation_line.split()[0]
        # 从文件中读取图像
        jpg = Image.open(r"./VOCdevkit/VOC2007/JPEGImages" + '/' + name + ".jpg")
        png = Image.open(r"./VOCdevkit/VOC2007/SegmentationClass" + '/' + name + ".png")

        if self.random_data:
            jpg, png = self.get_random_data(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))
        else:
            jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]), int(self.image_size[0])))

        # 从文件中读取图像
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        # 转化成one_hot的形式
        seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[1]), int(self.image_size[0]), self.num_classes+1))
        jpg = np.transpose(np.array(jpg), [2, 0, 1])/255

        return jpg, png, seg_labels


# DataLoader中collate_fn使用
def net_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels
