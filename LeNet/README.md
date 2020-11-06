# LeNet

---

### 目录
1. [实现的内容 Achievement](#实现的内容)
2. [注意事项](#所需环境)
3. [使用方式](#使用方式)
4. [网络架构](#网络架构)
### 实现的内容
- 使用CRFIA数据集实现图像分类

### 注意事项
- 如果没有数据集，需要先运行train.py文件自动下载数据集
- 网络需要GPU支持，若使用CPU，则注意删除代码中的‘cuda()’

### 使用方式
* （1）运行train.py文件即可自动下载数据集，并训练网络，生成的模型参数.pth存储在result文件中
* （3）找到一张图片，运行predict.py文件即可实现预测，输出照片的种类

### 网络架构
![photo](https://github.com/orange-eng/orange/raw/main/LeNet/LeNet_structure.png)
