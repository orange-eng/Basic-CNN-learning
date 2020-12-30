# tensorflow/pytorch_build_CNN
## 前言
* Tensorflow和Pytorch 是深度学习中最常用的框架之一，以后会经常使用到，所以在这里对tensorflow的构建CNN过程进行总结，以后搭模型会更方便
****

## 目录
1. [Tensorflow](#Tensorflow)
2. [Pytorch](#Pytorch)

## Tensorflow
这里以SRCNN为例，说明初始化参数的步骤
一般至少需要包含三个py文件，分别是主函数main，模型model以及底层数据处理工具utils(tools)
另外，还需要创建以下文件夹：test,train,result和checkpoint
接下来还是讲代码
```python

class SRCNN(object):

    #参数初始化
    def __init__(self,
                sess,
                image_size=33,
                label_size=21,
                batch_size=128,
                c_dim=1,
                checkpoint_dir=None,
                sample_dir=None):
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()    #这是搭建模型需要的函数
```
这里需要注意，最后一行build_model()是SRCNN的一个函数，主要用于构建模型基本参数，函数如下：
```python
#---------------------------------------------------初始化所有系数，w和b
def build_model(self):

    self.images = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.c_dim],name='images')
    self.labels = tf.placeholder(tf.float32,[None,self.label_size,self.label_size,self.c_dim],name='labels')
    self.weights = {
        'w1': tf.Variable(tf.random_normal([9,9,1,64],stddev=1e-3),name='w1'),
        'w2': tf.Variable(tf.random_normal([1,1,64,32],stddev=1e-3),name='w2'),
        'w3': tf.Variable(tf.random_normal([5,5,32,1],stddev=1e-3),name='w3'),
    }
    self.biases = {
        'b1': tf.Variable(tf.zeros([64]),name = 'b1'),
        'b2': tf.Variable(tf.zeros([32]),name = 'b2'),
        'b3': tf.Variable(tf.zeros([1]),name = 'b3')
    }
    self.pred = self.model()
    #Loss function(MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    self.saver = tf.train.Saver()   #an overview of variables, saving and restoring.

```
再写一个函数完成模型的结构
```python
#-----------------------------------------------------模型构建
    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images,self.weights['w1'],strides=[1,1,1,1],padding='VALID')+self.biases['b1'])        #输入是四位向量 kernel*kernel*input*output
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,self.weights['w2'],strides=[1,1,1,1],padding='VALID')+self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2,self.weights['w3'],strides=[1,1,1,1],padding='VALID')+self.biases['b3']
        return conv3
```

最后就是训练了
训练集和测试集的加载就不多说了， 感兴趣的朋友可以去看SRCNN模型，里面有详细加载H5模型的各种步骤
这里主要想展示几个最重要的训练函数
```python
def train(self, config):
    if config.is_train:
        input_setup(self.sess, config)
    else:
        nx, ny = input_setup(self.sess, config)
    if config.is_train:     
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
    train_data, train_label = read_data(data_dir)
#--------以上全部都是读取训练数据和测试数据，目的是得到train_data和train_label

    # Stochastic gradient descent with the standard backpropagation
    #一个优化器SGD
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
    #初始化所有参数
    tf.initialize_all_variables().run()
    counter = 0
    #记录模型运行时间
    start_time = time.time()

    if config.is_train:
        print("Training...")
        for ep in xrange(config.epoch):     #进入迭代过程
            # Run by batch images
            batch_idxs = len(train_data) // config.batch_size   #求出所有batch的个数，以batch为单位
            for idx in xrange(0, batch_idxs):
                batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
                counter += 1
                # 最重要的函数，得到loss函数
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                #feed_dict={y:3}表示把3赋值给y
                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                    % ((ep+1), counter, time.time()-start_time, err))

    else:
        print("Testing...")
        #预测结果
        result = self.pred.eval({self.images: train_data, self.labels: train_label})

```

## Pytorch
这里以最简单的AlexNet为例子，说明Pytorch的训练过程
一般至少要包含三个文件:模型Net,训练train和预测predict
此外，还可以创建文件夹train和result用于存放数据
接下来开始讲代码

```python
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
```
如果想观察模型架构，可以运行如下代码：
```python
model=AlexNet()   
print(model)
```

训练函数
```python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),     #随机裁剪到224
                                 transforms.RandomHorizontalFlip(),     #随机水平竖直翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),           # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
image_path = path  # flower data set path
train_dataset = datasets.ImageFolder(root=image_path + '\\flower_data\\train',
                                     transform=data_transform["train"])
train_num = len(train_dataset)      #train_set总共3306张图片
print(train_num)
#生成一个索引文件json
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)       
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

```
载入数据
```python
    #--------------------载入数据----------------------------------
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=image_path + '\\flower_data\\val',
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=0)
```

模型构建
```python
#------------------------网络构建------------------------------
net=AlexNet(num_class=5,init_Weight=True)       #初始化权重设置为Ture
net.to(device)

loss_function = nn.CrossEntropyLoss()           #交叉熵
loss_function.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.0002)

save_path = path + '\\result\AlexNet.pth'
```
迭代过程
```python
for epoch in range(3):
    #--------------------------train-------------------
    net.train()
    running_loss=0
    t1=time.perf_counter()
    for step,data in enumerate(train_loader,start=0):
        images,labels = data
        optimizer.zero_grad()                               #初始化为0

        outputs=net(images.to(device))                      #用GPU
        loss=loss_function(outputs,labels.to(device))                  #计算损失函数
        loss.backward()                                     #反向传播
        optimizer.step()

        #---------------打印训练进度数据-------------------------
        running_loss +=loss.item()
        rate = (step+1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)                           #统计计算时间

    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:                    #已经分类好的样本
            val_images, val_labels = val_data

            outputs = net(val_images.to(device))

            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()        #每迭代一次之后，计算准确率
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')
```

预测部分
```python

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# load image
img = Image.open(path+"/img/1.png")
plt.imshow(img)
plt.show()
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = AlexNet(num_class=5)
# load model weights
model_weight_path = path+"/result/AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()


```
