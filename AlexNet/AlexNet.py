import torch.nn as nn
import torch.nn.functional as F
import torch


class AlexNet(nn.Module):
    def __init__(self,num_class=1000,init_Weight=False):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2), # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),                                       # output[48, 27, 27]
            nn.Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2),         # output[128,27,27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),                                       # output[128,13,13]
            nn.Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=1),         # output[192,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),         # output[192,13,13]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=3,stride=1,padding=1),         # output[128,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),                                       # output[128,6,6]
        )
        self.classifier=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128*6*6,out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048,out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_class),
        )
        if init_Weight:
            self._initialize_weights()
    #前向传播过程，添加分类器
    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():        #继承了nn.module，会返回一个迭代器，遍历每一个模块
            if isinstance(m, nn.Conv2d):        #判断m是否是给定类型nn.Conv2d
                #如果是Conv2d，就是用kaiming_normal_对权重进行初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    #如果偏置不为空，就用0进行初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #如果是全连接层，使用normal进行初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model=AlexNet()   
print(model)

