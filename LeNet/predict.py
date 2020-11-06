import torch
import torchvision.transforms as transforms
from PIL import Image
from LeNet import LeNet
import numpy
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

transform = transforms.Compose(
    [transforms.Resize((32, 32)),           #模型的输入是32*32
     transforms.ToTensor(),                 #转化为tensor才可以
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.cuda()

net.load_state_dict(torch.load(path+'./result/Lenet.pth'))


image = Image.open(path+'./img/2.png')
image = transform(image)  # [C, H, W] channel hight width
image = torch.unsqueeze(image, dim=0)  # [N, C, H, W] 再加一个维度 变成batch channel hight width
image=image.cuda()

with torch.no_grad():
    outputs = net(image)
    outputs=outputs.cpu()   #需要把CUDA格式转化为CPU格式才能训练和预测
    predict = torch.max(outputs, dim=1)[1]
print(classes[int(predict)])
