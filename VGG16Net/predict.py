import torch
from VGG16Net import vgg
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt 
import json

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

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
    json_file = open(path+'./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
# create model
model = vgg(num_classes=5)
# load model weights
model_weight_path = path+"/result/VGG16Net.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
