import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from AlexNet import AlexNet
import json
import time

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
with open(path+'\\class_indices.json', 'w') as json_file:
    json_file.write(json_str)

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
'''
#查看图片样本
test_data_iter = iter(validate_loader)
test_image, test_label = test_data_iter.next()

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
imshow(utils.make_grid(test_image))
'''
#------------------------网络构建------------------------------
net=AlexNet(num_class=5,init_Weight=True)       #初始化权重设置为Ture
net.to(device)

loss_function = nn.CrossEntropyLoss()           #交叉熵
loss_function.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.0002)

save_path = path + '\\result\AlexNet.pth'
best_acc=0
for epoch in range(1):
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




