import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from nets.net_unetpp import MyNet
from nets.net_segnet import SegNet
from nets.net_unet import Unet
from utils.loss import CE_Loss, Dice_loss
from utils.metrics import f_score
from torch.utils.data import DataLoader
from utils.dataloader import net_dataset_collate, NetDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 训练
def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, aux_branch):
    '''epoch_size:训练数据生成器最多生成多少批，epoch_size_val:验证数据生成器最多生成多少批，gen:训练数据生成器，genval:验证数据生成器
    epoch:第几轮，Epoch:总轮数'''
    net = net.train()

    total_loss = 0
    total_f_score = 0
    val_toal_loss = 0
    val_total_f_score = 0

    start_time = time.time()

    # tqdm是一个显示循环的进度条的库
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        # 控制训练数据生成器迭代
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:   # 一旦超过最大批次就停止生成，因为剩余的数据不能成一批
                break
            imgs, pngs, labels = batch

            with torch.no_grad():        # 使用 with torch.no_grad():强制之后的内容不进行计算图构建
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.to(device)
                    pngs = pngs.to(device)
                    labels = labels.to(device)

            optimizer.zero_grad()
            # 计算损失
            if aux_branch:
                aux_outputs, outputs = net(imgs)
                aux_loss  = CE_Loss(aux_outputs, pngs, num_classes=NUM_CLASSES)
                main_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                loss      = aux_loss + main_loss
                if dice_loss:
                    aux_dice  = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + aux_dice + main_dice

            else:
                outputs = net(imgs)
                loss    = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

            # 计算metrics
            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()
            
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    net.eval()   # 验证时需要eval模式
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.to(device)
                    pngs = pngs.to(device)
                    labels = labels.to(device)

                if aux_branch:
                    aux_outputs, outputs = net(imgs)
                    aux_loss  = CE_Loss(aux_outputs, pngs, num_classes = NUM_CLASSES)
                    main_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    val_loss  = aux_loss + main_loss
                    if dice_loss:
                        aux_dice  = Dice_loss(aux_outputs, labels)
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + aux_dice + main_dice
                else:
                    outputs  = net(imgs)
                    val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        val_loss  = val_loss + main_dice

                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    net.train()
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_toal_loss/(epoch_size_val+1)))


if __name__ == "__main__":          # 这行代码表示：当该文件被直接运行时，以下代码块将被运行，当该文件是被导入时，以下代码块不被运行
    inputs_size = [512, 512, 3]     # 输入网络的图片尺寸
    log_dir = "logs/"               # 权重存储目录
    NUM_CLASSES = 21                # 分类数
    dice_loss = False               # 是否使用diceloss，通常医学图像用的多
    pretrained = True               # 是否使用预训练权重
    backbone = "resnet50"           # 骨干网络的选择
    aux_branch = False              # 是否使用辅助分支帮助训练
    downsample_factor = 16          # backbone输出的图片是原图下采样多少倍
    Cuda = True

    # model = MyNet(num_classes=NUM_CLASSES, backbone=backbone, down_tate=downsample_factor,
    #                pretrained=pretrained, aux_branch=aux_branch).train()
    # model = Unet(num_classes=NUM_CLASSES, in_channels=inputs_size[-1], pretrained=pretrained).train()  # 训练unet
    model = SegNet(input_channels=3, output_channels=21)
    # 导入以及训练好的权重
    # model_path = r"model_data/backbone_82.56.pth"
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()               # 按state_dict导入权重
    # pretrained_dict = torch.load(model_path)
    #
    # #按节点匹配权重
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)
    # print('Finished!')


    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.to(device)

    # 打开训练数据的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt", "r") as f:
        train_lines = f.readlines()

    # 打开验证数据的txt
    with open(r"VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", "r") as f:
        val_lines = f.readlines()
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    '''
    if True:
        lr = 1e-2
        Init_Epoch = 0
        Interval_Epoch = 50
        Batch_size = 8
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)

        train_dataset = NetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = NetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=6, pin_memory=True,
                                drop_last=True, collate_fn=net_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=net_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Interval_Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Interval_Epoch,Cuda,aux_branch)
            lr_scheduler.step()
        '''
    if True:
        lr = 1e-2
        Interval_Epoch = 0
        Epoch = 200
        Batch_size = 8
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        train_dataset = NetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = NetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=net_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True,
                                drop_last=True, collate_fn=net_dataset_collate)

        epoch_size      = max(1, len(train_lines)//Batch_size)
        epoch_size_val  = max(1, len(val_lines)//Batch_size)

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda,aux_branch)
            lr_scheduler.step()

