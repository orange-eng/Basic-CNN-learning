# orange
## 前言
* 该markdown文档里面包含了自己写代码的时候遇到的许多的问题和笔记。
****

# 笔记汇总

## 编程技巧

### H5文件
**h5文件中有两个核心的概念：组“group”和数据集“dataset”。 一个h5文件就是 “dataset” 和 “group” 二合一的容器。**

- dataset ：简单来讲类似数组组织形式的数据集合，像 numpy 数组一样工作，一个dataset即一个numpy.ndarray。具体的dataset可以是图像、表格，甚至是pdf文件和excel。
- group：包含了其它 dataset(数组) 和 其它 group ，像字典一样工作。

   一个h5文件被像linux文件系统一样被组织起来：dataset是文件，group是文件夹，它下面可以包含多个文件夹(group)和多个文件(dataset)。

**使用python对h5文件进行操作**

- 写H5文件

```python
# Writing h5

import h5py
import numpy as np
# mode可以是"w",为防止打开一个已存在的h5文件而清除其数据,故使用"a"模式
with h5py.File("animals.h5", 'a') as f:
    f.create_dataset('animals_included',data=np.array(["dogs".encode(),"cats".encode()])) # 根目录下创建一个总览介绍动物种类的dataset,字符串应当字节化
    dogs_group = f.create_group("dogs") # 在根目录下创建gruop文件夹:dogs
    f.create_dataset('cats',data = np.array(np.random.randn(5,64,64,3))) # 根目录下有一个含5张猫图片的dataset文件
    dogs_group.create_dataset("husky",data=np.random.randn(64,64,3)) # 在dogs文件夹下分别创建两个dataset,一张哈士奇图片和一张柴犬的图片
    dogs_group.create_dataset("shiba",data=np.random.randn(64,64,3))
```
- 读取H5文件

```python
with h5py.File('animals.h5','r') as f:
    for fkey in f.keys():
        print(f[fkey], fkey)

    print("======= 优雅的分割线 =========")
    '''
    结果：
    <HDF5 dataset "animals_included": shape (2,), type "|S4"> animals_included
	<HDF5 dataset "cats": shape (5, 64, 64, 3), type "<f8"> cats
	<HDF5 group "/dogs" (2 members)> dogs
	'''

    dogs_group = f["dogs"] # 从上面的结果可以发现根目录/下有个dogs的group,所以我们来研究一下它
    for dkey in dogs_group.keys():
        print(dkey, dogs_group[dkey], dogs_group[dkey].name, dogs_group[dkey].value)
```
这里还有一个技巧，就是如果需要直接读取H5文件中的数据，只能得到dataset里面的数据，可以采用如下命令来实现：
```python
#此时h5文件中包含两个dataset: data和llabel
with h5py.File(name=path,mode='r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data,label
```

### print函数
```python
print('%s and %s.'%(line,line[0]))
print("{0} and {1}.".format(line,line[0]))
```
### tf.session函数
```python
A class for running TensorFlow operations.

tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Evaluate the tensor `c`.
print(sess.run(c)) # prints 30.0
```

### Python 数据结构
#### 列表
* 在python中，矩阵是按照行来存储的，如果要读取列向量，则需要简单的索引。如下
```python
M = [
  [1,2,3],
  [4,5,6],
  [7,8,9]
]
col2 = [ row[1] for row in M]
#输出为
[2, 5, 8]
```

### 初始化所有参数

```python
#----------------------------------------一种初始化所有参数的方法
flags = tf.app.flags
flags.DEFINE_integer("epoch",3,"Number of epoch[3]")
flags.DEFINE_integer("batch_size",128,"the size of batch [128]")
flags.DEFINE_integer("image_size",33,"the size of image [33]")
flags.DEFINE_integer("label_size",21,"The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS
```


调用的方法也十分简单
```python
print(FLAGS.learning_rate)
```

### os库的使用
#### os.path.join()： 将多个路径组合后返回
```python
Path1 = 'a'
Path2 = 'b'
Path3 = 'c'

Path_n = Path1 + Path2 + Path3
Path_n1 = os.path.join(Path1,Path2,Path3)
print ('Path_n = ',Path_n)
print ('Path_n1 = ',Path_n1)

#输出为
Path_n = abc
Path_n1 = a\b\c
```
#### os.listdir()用于返回一个由文件名和目录名组成的列表，需要注意的是它接收的参数需要是一个绝对的路径
```python
import os
path = '/home/python/Desktop/'
for i in os.listdir(path):
    print(i)

#输出为该绝对路径下面的所有文件名

```

#### os.sep:  python是跨平台的。在Windows上，文件的路径分隔符是’’，在Linux上是’/’。为了让代码在不同的平台上都能运行，那么路径应该写’‘还是’/'呢？使用os.sep的话，就不用考虑这个了，os.sep根据你所处的平台，自动采用相应的分隔符号。

#### glob.glob()将目录下的文件全部读取了出来

## 神经网络的反向传播算法
推导过程请看链接
https://blog.csdn.net/qq_29407397/article/details/90599460
卷积神经网络反向传播
https://blog.csdn.net/weixin_40446651/article/details/81516944

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

## 将batch_size的作用

**首先，为什么需要有 Batch_Size 这个参数**

Batch 的选择，首先决定的是下降的方向。如果数据集比较小，完全可以采用全数据集 （ Full Batch Learning ）的形式，这样做至少有 2 个好处：其一，由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。其二，由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。

对于更大的数据集，以上 2 个好处又变成了 2 个坏处：其一，随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。其二，以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来 RMSProp 的妥协方案。

**既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？**

所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。这就是在线学习（Online Learning）。线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，难以达到收敛。

**可不可以选择一个适中的 Batch_Size 值呢？**

当然可以，这就是批梯度下降法（Mini-batches Learning）。因为如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的。

**在合理范围内，增大 Batch_Size 有何好处？**

内存利用率提高了，大矩阵乘法的并行化效率提高。
跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。

**盲目增大 Batch_Size 有何坏处？**

内存利用率提高了，但是内存容量可能撑不住了。
跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

**总结**

- Batch_Size 太小，算法在 200 epoches 内不收敛。
- 随着 Batch_Size 增大，处理相同数据量的速度越快。
- 随着 Batch_Size 增大，达到相同精度所需要的 epoch 数量越来越多。
- 由于上述两种因素的矛盾， Batch_Size 增大到某个时候，达到时间上的最优。
- 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到某些时候，达到最终收敛精度上的最优。

## Batch Normalization

**一、简介**

BN是由Google于2015年提出，这是一个深度神经网络训练的技巧，它不仅可以加快了模型的收敛速度，而且更重要的是在一定程度缓解了深层网络中“梯度弥散（特征分布较散）”的问题，从而使得训练深层网络模型更加容易和稳定。所以目前BN已经成为几乎所有卷积神经网络的标配技巧了。
从字面意思看来Batch Normalization（简称BN）就是对每一批数据进行归一化，确实如此，对于训练中某一个batch的数据{x1,x2,…,xn}，注意这个数据是可以输入也可以是网络中间的某一层输出。在BN出现之前，我们的归一化操作一般都在数据输入层，对输入的数据进行求均值以及求方差做归一化，但是BN的出现打破了这一个规定，我们可以在网络中任意一层进行归一化处理，因为我们现在所用的优化方法大多都是min-batch SGD，所以我们的归一化操作就成为Batch Normalization。

**2.BN的作用**

但是我们以前在神经网络训练中，只是对输入层数据进行归一化处理，却没有在中间层进行归一化处理。要知道，虽然我们对输入数据进行了归一化处理，但是输入数据经过σ ( W X + b ) σ(WX+b)σ(WX+b)这样的矩阵乘法以及非线性运算之后，其数据分布很可能被改变，而随着深度网络的多层运算之后，数据分布的变化将越来越大。如果我们能在网络的中间也进行归一化处理，是否对网络的训练起到改进作用呢？答案是肯定的。
这种在神经网络中间层也进行归一化处理，使训练效果更好的方法，就是批归一化Batch Normalization（BN）。BN在神经网络训练中会有以下一些作用：
- 加快训练速度
- 可以省去dropout，L1, L2等正则化处理方法
- 提高模型训练精度

**BN流程**

* 1.求每一个小批量训练数据的均值
* 2.求每一个小批量训练数据的方差
* 3.使用求得的均值和方差对该批次的训练数据做归一化，获得0-1分布。其中ε εε是为了避免除数为0时所使用的微小正数。
* 4.尺度变换和偏移：由于归一化后的x被限制在正态分布下，使得网络的表达能力下降。

**BN的本质就是利用优化变一下方差大小和均值位置，使得新的分布更切合数据的真实分布，保证模型的非线性表达能力。BN的极端的情况就是这两个参数等于mini-batch的均值和方差，那么经过batch normalization之后的数据和输入完全一样，当然一般的情况是不同的。
BN在深层神经网络的作用非常明显：若神经网络训练时遇到收敛速度较慢，或者“梯度爆炸”等无法训练的情况发生时都可以尝试用BN来解决。同时，常规使用情况下同样可以加入BN来加速模型训练，甚至提升模型精度。**

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

**6)问：python报错解决方法：module 'scipy.misc' has no attribute 'imread'**

**答： 在该python环境中，安装Pillow即可；或是scipy1.3.0的版本降级到scipy==1.2.1才行**

**7)问：数字后面加.表示什么意思呢？**

**答： 数字后面加.表示该数值为浮点型float**

**8)问：色彩空间YCbCr指的是什么呢？**

**答： YCbCr或Y'CbCr有的时候会被写作：YCBCR或是Y'CBCR，是色彩空间的一种，通常会用于影片中的影像连续处理，或是数字摄影系统中。Y'为颜色的亮度(luma)成分、而CB和CR则为蓝色和红色的浓度偏移量成份。Y'和Y是不同的，而Y就是所谓的亮度(luminance)，表示光的浓度且为非线性，使用伽马修正(gamma correction)编码处理；**

**YCbCr其中Y是指亮度分量，Cb指蓝色色度分量，而Cr指红色色度分量。人的肉眼对视频的Y分量更敏感，因此在通过对色度分量进行子采样来减少色度分量后，肉眼将察觉不到的图像质量的变化。主要的子采样格式有 YCbCr 4:2:0、YCbCr 4:2:2 和 YCbCr 4:4:4。**

**4:2:0表示每4个像素有4个亮度分量，2个色度分量 (YYYYCbCr），仅采样奇数扫描线，是便携式视频设备（MPEG-4）以及电视会议（H.263）最常用格式；4：2：2表示每4个像素有4个亮度分量，4个色度分量（YYYYCbCrCbCr），是DVD、数字电视、HDTV 以及其它消费类视频设备的最常用格式；4：4：4表示全像素点阵(YYYYCbCrCbCrCbCrCbCr），用于高质量视频应用、演播室以及专业视频产品。**
