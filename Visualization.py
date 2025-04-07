import torch
from nets.net_fcn import MyNet
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MyNet(num_classes=21, backbone="resnet50", down_tate=16,
               aux_branch=True, pretrained=True).train().to(device)

summary(model, (3, 512, 512))