import torch.nn as nn
import torch
import torchvision.models.resnet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                                kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features = out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                                kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.downsample = downsample
    def forward(self,x):
        plus = x
        if self.downsample is not None:
            plus = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #两次卷积之后，图像的尺寸并没有发生改变
        out = out + plus            #残差部分的叠加
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,block_num,num_classes=1000):
        #block就是定义的残差结构
        super(ResNet,self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.__make_layer(block=block,channel=64,block_num=block_num[0],stride=1)
        self.layer2 = self.__make_layer(block=block,channel=128,block_num=block_num[1],stride=2)
        self.layer3 = self.__make_layer(block=block,channel=256,block_num=block_num[2],stride=2)
        self.layer4 = self.__make_layer(block=block,channel=512,block_num=block_num[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))      #output size = (1,1)
        self.fc = nn.Linear(in_features=512,out_features=num_classes)

    def __make_layer(self,block,channel,block_num,stride):
        downsample = None
        if stride != 1 or self.in_channel != block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)) 
        
        layers = []
        layers.append(block(self.in_channel,channel,downsample=downsample,stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel,stride=1))
        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def resnet34(num_classes = 1000):
    return ResNet(block=BasicBlock,block_num=[3,4,6,3],num_classes=num_classes)

net = resnet34()
print(net)







