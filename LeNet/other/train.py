import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.utils.data as utils_data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t
from LeNet import LeNet

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
#--------------------------LeNet--------------------------------
gpus = [0]                              #使用哪几个GPU进行训练，这里选择0号GPU
cuda_gpu = t.cuda.is_available()   #判断GPU是否存在可用
print(cuda_gpu)
net=LeNet()                             #载入网络
if(cuda_gpu):
    #net = t.nn.DataParallel(net, device_ids=gpus).cuda()   #将模型转为cuda类型
    net=net.cuda()
#---------------------载入数据---------------------------------------
show=ToPILImage()                                       #可以把Tensor转化成image
transform_CIFAR=transforms.Compose([
    transforms.ToTensor(),                              #转化为Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))   #归一化
])
#-------------------训练集-----------------------------
trainset=tv.datasets.CIFAR10(
    root=path+'\\trainset',
    train=True,
    download=True,
    transform=transform_CIFAR
)
trainloader=utils_data.DataLoader(
    trainset,
    batch_size=4,                   #每次输入的数据行数
    shuffle=True,                   #每次训练要给数据洗牌
    num_workers=0                   #有两个子进程来导入数据
)
#-----------------测试集---------------------------------
testset=tv.datasets.CIFAR10(
    root=path+'\\trainset',
    train=False,
    download=True,
    transform=transform_CIFAR
)
testloader=utils_data.DataLoader(
    trainset,
    batch_size=4,                   #每次输入的数据行数
    shuffle=True,                   #每次训练要给数据洗牌
    num_workers=0                   #有两个子进程来导入数据
)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(CIDAR_data,label)=trainset[100]            #CIFIA数据集返回data和label两个参数
#-----------------------------可视化---------------------------
'''
print(classes[label])
picture1=show((CIDAR_data+1)/2).resize((100,100))
picture1=show((CIDAR_data+1)/2)
print(picture1.size)
plt.imshow(picture1)
plt.show()
'''
#---------------------------定义优化器和损失函数--------------
criterion=nn.CrossEntropyLoss()         #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#------------------------------训练网络----------------------------
for epoch in range(1):
    running_loss=0.0
    
    for i,data in enumerate(trainloader,0):
        
        #输入数据
        inputs,labels=data
        inputs,labels=Variable(inputs).cuda(),Variable(labels).cuda()
        #梯度清零
        optimizer.zero_grad()
        #forward+backward
        outputs=net(inputs).cuda()
        criterion.cuda()
        loss=criterion(outputs,labels)
        loss.backward()
        #更新参数
        optimizer.step()
        #打印log信息
        running_loss+=loss.data
        if(i%2000 == 1999):          #2000个batch打印一次训练状态
            print('[%d,%d] loss: %.3f' \
                % (epoch+1,i+1,running_loss/2000))
            running_loss=0
        
    print("finish training")

#-------------------------测试-------------------------------------
corrent=0   #预测正确的图片数
total=0     #总图片数目
for data in testloader:
    images,labels = data.cuda()
    outputs_test=net(Variable(images)).cuda()
    _,predicted=t.max(outputs_test.data,1).cuda()
    total+=labels.size(0)
    corrent+=(predicted==labels).sum()
print('测试集中准确率:%d %%' % (100*corrent/total))
