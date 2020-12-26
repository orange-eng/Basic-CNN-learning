# orange
## 前言
* 该markdown文档里面包含了自己写代码的时候遇到的许多的问题。
****

# 笔记汇总
## 激活函数
“激活函数”能分成两类——“饱和激活函数”和“非饱和激活函数”

sigmoid和tanh是“饱和激活函数”，而ReLU及其变体则是“非饱和激活函数”。使用“非饱和激活函数”的优势在于两点：

    1.首先，“非饱和激活函数”能解决所谓的“梯度消失”问题。
    2.其次，它能加快收敛速度。
    Sigmoid函数需要一个实值输入压缩至[0,1]的范围
    σ(x) = 1 / (1 + exp(−x))
    tanh函数需要讲一个实值输入压缩至 [-1, 1]的范围
    tanh(x) = 2σ(2x) − 1

### ReLU
    ReLU函数代表的的是“修正线性单元”，它是带有卷积图像的输入x的最大函数(x,o)。ReLU函数将矩阵x内所有负值都设为零，其余的值不变。ReLU函数的计算是在卷积之后进行的，因此它与tanh函数和sigmoid函数一样，同属于“非线性激活函数”。这一内容是由Geoff Hinton首次提出的。
### ELUs
    ELUs是“指数线性单元”，它试图将激活函数的平均值接近零，从而加快学习的速度。同时，它还能通过正值的标识来避免梯度消失的问题。根据一些研究，ELUs分类精确度是高于ReLUs的。下面是关于ELU细节信息的详细介绍：

### Leaky ReLUs
    ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以表示为：
## 人脸数据集链接
https://www.cnblogs.com/haiyang21/p/11208293.html

### 计算Leaky ReLU激活函数
```python
    tf.nn.leaky_relu(
    features,
    alpha=0.2,
    name=None )
```
    参数： features:一个Tensor,表示预激活
    alpha:x<0时激活函数的斜率
    ame:操作的名称（可选）
    返回值：激活值
    数学表达式： y = max(0, x) + leak*min(0,x)
    优点：
    1.能解决深度神经网络（层数非常多）的“梯度消失”问题，浅层神经网络（三五层那种）才用sigmoid 作为激活函数。
    2.它能加快收敛速度。



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
****

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
****
## 3. 代码参数
**1)问：args和kwargs在代码中是什么意思呢?**

**答： args和kwargs一般是用在函数定义的时候。
二者的意义是允许定义的函数接受任意数目的参数。
也就是说我们在函数被调用前并不知道也不限制将来函数可以接收的参数数量。
在这种情况下我们可以使用args和kwargs。**

**2)问：网络结构中，发现output的尺寸与理论值不同怎么办？**

**答： 首先要分析哪个参数不同（batch_size,channels,height,width）。最常见的是height和width出现异常，此时需要检查每一层对padding的设置，因为这会直接影响输出的大小。可以采用padding='SAME'的方式，使得输出层与输入层尺度保持不变**

****
## 4. 知识点（推荐的博客）
**1)问：mAP是什么？**

**答： 参考连接：https://github.com/XifengGuo/CapsNet-Keras/issues/7**


**1)问：Batch Normalization详解**

**答： 参考连接：https://blog.csdn.net/qq_37541097/article/details/104434557  在我们训练完后我们可以近似认为我们所统计的均值和方差就等于我们整个训练集的均值和方差。然后在我们验证以及预测过程中，就使用我们统计得到的均值和方差进行标准化处理。**

****
## 5. 常见报错
**1)问：TypeError: not all arguments converted during string formatting**

**答： 此时往往有单词拼写错误，认真检查参数和单词拼写**


**2)问：RuntimeError: CUDA out of memory. Tried to allocate 98.00 MiB**

**答： 这是用于batch_size设置太大，导致CUDA无法计算。将batch_size调小一点即可**

**3)问：我的git clone之后为什么没有.git文件呢**

**答： 需要点击“查看”，并显示隐藏文件，即可看到git文件**

**4)问：如何把彩色图像变为灰度图像**
```python
from PIL import Image
import numpy as np
a=np.array(Image.open(r"C:\Users\23263\Desktop\2\1.jpg").convert('L'))
b=255-a
im=Image.fromarray(b.astype('uint8'))
im.save(r"C:\Users\23263\Desktop\2\2.jpg")
```
**5)问：图像之间的格式转化**
```python

import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
import tensorflow as tf
imagePath1=path+"/img/1.png"
imagePath2=path+"/img/2.png"

#--------------------------------------------------使用PIL读取图像
img1 = Image.open(imagePath1)

#--------------------------------------------------使用opencv-python读取图像
img_cv2_color = cv2.imread(imagePath1)  # cv2读入的图像默认是uint8格式的numpy.darray, BGR通道
print(type(img_cv2_color))
img_cv2_gray = cv2.imread(imagePath1, 0)  # img_cv2_gray.shape (512,512)
img = Image.fromarray(img_cv2_color)    #把numpy.ndarry格式变成PIL.Image格式

#--------------------------------------------------使用matplotlib读取图像
img_matplot = mpimg.imread(imagePath1)
print(type(img_matplot))
plt.imshow(img_matplot)
plt.show()

#--------------------------------------------------PIL+matplotlib
# 统一使用plt进行显示, 不管是plt还是cv2.imshow, 在python中只认numpy.ndarray

img = Image.open(imagePath1)  # PIL读入的图像自然就是uint8格式
#img.show()
img = np.array(img)  # 获得numpy对象, np.ndarray, RGB通道
plt.imshow(img)
plt.show()

#---------------------------------------------------skimage+matplotlib

from skimage import io
img = io.imread(imagePath1)     #格式为numpy.ndarry
print(type(img))
plt.imshow(img)
plt.show()


#----------------------------------------------------PIL.Image与numpy.ndarry之间的互换
img_pil = Image.open(imagePath1)   # PIL读入的图像自然就是uint8格式
a = np.array(img_pil)  # PIL.Image 转换成 numpy.darray

# 先把numpy.darray转换成np.unit8, 确保像素值取区间[0,255]内的整数
# 灰度图像需保证numpy.shape为(H,W)，不能出现channels，可通过执行np.squeeze()剔除channels；
# 彩色图象需保证numpy.shape为(H,W,3)
a = a.astype(np.uint8)  # a.astype('uint8')  # a = np.uint8(a)
# 再转换成PIL Image形式
img = Image.fromarray(a)  # numpy.darray 转换成 PIL.Image
img.show()

#---------------------------------------------------------array和tensor之间的转换
# 主要是两个方法：
# 1.数组转tensor:数组a,  tensor_a=tf.convert_to_tensor(a)
# 2.tensor转数组：tensor b, array_b=b.eval()
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=tf.constant(a)                                             #一个tensor常量
print(type(b))

array_b = []
with tf.Session() as sess:
    print (b)
    for x in b.eval():      #b.eval()就得到tensor的数组形式
        print (x)
        array_b.append(x)
    print ('a是数组',a)
    print(array_b)
    tensor_a=tf.convert_to_tensor(a)
    
```
