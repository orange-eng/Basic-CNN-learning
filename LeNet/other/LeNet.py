import torchvision as tv
import torchvision.transforms as transforms
#--------------------------LeNet--------------------------------
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t
class LeNet(nn.Module):
    def __init__(self): #nn.Module子类在构造函数中执行父类的构造函数
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)             #输入通道为1，输出为6，kernel_size=5
        self.conv2=nn.Conv2d(6,16,5)            #输入通道为6，输出为16，kernel_size=5
        self.fc1=nn.Linear(16*5*5,120)          #输入为16*5*5的单列向量，输出为120
        self.fc2=nn.Linear(120,84)              #输入为120的单列向量，输出为84
        self.fc3=nn.Linear(84,10)               #输入为84的单列向量，输出为10

    def forward(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))     #kernel_size=2
        x=F.max_pool2d(F.relu(self.conv2(x)),2)         #kernel_size=2
        #reshape -1表示自适应
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
'''
net=LeNet()
print(net)             #输出网络结构

params=list(net.parameters())
print(len(params))      #输出层的长度

for name,parameters in net.named_parameters():
    print(name,':',parameters.size())   #打印每一层的结构
#-------------------------
input = Variable(t.randn(1,1,32,32))    #随机生成一个1*32*32的数据用于实验
output=net(input)           
print(output.size())
target=t.rand(1,10)
#target= Variable(t.arange(0,10))
print(target.size())
criterion = nn.MSELoss()                #计算均方误差
loss=criterion(output,target)
print(loss)
'''