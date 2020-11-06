# AlexNet

---

### 目录
1. [实现的内容 Achievement](#实现的内容)
2. [数据集加载](#注意事项)
3. [使用方式](#使用方式)
4. [网络架构](#网络架构)
### 实现的内容
#### LeNet
- 使用花数据集，实现5种花的图像分类

### 数据集加载
#### 该文件夹是用来存放训练样本的目录
##### 使用步骤如下：
* （1）打开"flower_data"
* （2）打开flower_link.txt文档，复制网址到浏览器会自动进行下载花分类数据集
* （3）解压数据集到flower_data文件夹下
* （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val    
  （不要重复使用该脚本，否则训练集和验证集会混在一起，flower_data文件夹结构如下）   
  |—— flower_data   
  |———— flower_photos（解压的数据集文件夹，3670个样本）  
  |———— train（生成的训练集，3306个样本）  
  |———— val（生成的验证集，364个样本）
### 使用方式
* （1）先按照“数据集加载”操作生成数据集并分类
* （2）运行train.py文件即可训练网络，生成的模型参数.pth存储在result文件中
* （3）找到一张花的图片，运行predict.py文件即可实现预测，输出花的种类
### 网络架构

![photo](https://github.com/orange-eng/orange/raw/main/AlexNet/AlexNet_Structure.png)
