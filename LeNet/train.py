import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt 
from LeNet import LeNet
import torchvision
import torch
import torch.nn as nn

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   #标准化output=(input-0.5)/0.5

# 50000张训练图片
train_set = torchvision.datasets.CIFAR10(root=path+'\\trainset', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                          shuffle=False, num_workers=0)

# 10000张验证图片
val_set = torchvision.datasets.CIFAR10(root=path+'\\trainset', train=False,
                                       download=False, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                         shuffle=False, num_workers=0)
# val_data_iter = iter(val_loader)
# val_image, val_label = val_data_iter.next()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize input=output*0.5+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# print labels
print(' '.join('%5s' % classes[val_label[j]] for j in range(4)))
# show images
imshow(torchvision.utils.make_grid(val_image))
'''
net = LeNet()
net.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
#-----------------训练过程---------------------------------
for epoch in range(5):
    running_loss=0.0
    for step,data in enumerate(train_loader,start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()   
        #forward+backward+optimize
        outputs=net(inputs)
        outputs=outputs.cuda()

        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data

        if step % 1000 == 999:    # print every 500 mini-batches
            #with torch.no_grad():
            for data_test in val_loader:
                val_image,val_labels=data_test
                val_image=val_image.cuda()
                val_labels=val_labels.cuda()
                outputs = net(val_image)  
                outputs=outputs.cuda()

            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = (predict_y == val_labels).sum().item() / val_labels.size(0)      #预测正确时，并求和，item返回该数值
            print((predict_y == val_labels).sum().item())    
            print(val_labels.size(0))
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                (epoch + 1, step + 1, running_loss / 1000, accuracy))
            running_loss = 0.0   
print('Finished Training')

#-------------------存储模型-----------------------------------------
save_path = path+'./result/Lenet.pth'
torch.save(net.state_dict(), save_path)












