# orange
## 前言
* 该markdown文档里面包含了自己写代码的时候遇到的许多的问题。


# 问题汇总
## 1、路径问题
**问：在代码里面，怎么用相对路径呢？（使用相对路径的话，别人copy我的代码就可以直接使用）？**

**答：在代码开头加上如下几句**

```python
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
```

**之后使用path就是当前文件的路径，十分方便。如果想调用其他文件，采用如下写法即可**
```python
root=path+'\\trainset'
```
**root就是该路径下，trainset文件中的内容路径**


## 2、GPU问题
**问：我好像没有在用gpu进行训练啊，怎么看是不是用了GPU进行训练?**

**答：查看是否使用GPU进行训练一般使用NVIDIA在命令行的查看命令，如果要看任务管理器的话，请看性能部分GPU的显存是否利用，或者查看任务管理器的Cuda，而非Copy。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013234241524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc5MTk2NA==,size_16,color_FFFFFF,t_70#pic_center)
**问：怎么使用torch运行GPU呢？**

**答：首先要判断是否有GPU可以使用。可以运行如下代码**
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```
**之后在跑网络时，需要在net,input,labels,output,loss等参数后面加上.cuda()或是to(device)才可以。例如下面的代码即可调用GPU**
```python
#example1
net=AlexNet(num_class=5,init_Weight=True)       #初始化权重设置为Ture
net.to(device)
loss_function = nn.CrossEntropyLoss()           #交叉熵
loss_function.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.0002)

#example2
for step,data in enumerate(train_loader,start=0):
    images,labels = data
    outputs=net(images.to(device))                      #用GPU
    loss=loss_function(outputs,labels.to(device))       #计算损失函数
```
