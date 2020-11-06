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

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
'''
input1=torch.rand([32,3,224,224])
model=AlexNet()   
print(model)
'''

