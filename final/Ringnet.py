import urllib
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

dev = "cpu"
if torch.cuda.is_available():  
  dev = "cuda:0"   
device = torch.device(dev)  

res_out_size = 2048
fc1_out_size = 512
fc2_out_size = 512
fc3_out_size = 159

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x

class Regression(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(res_out_size, fc1_out_size, bias=True)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(p=0.2)

    self.fc2 = nn.Linear(fc1_out_size, fc2_out_size, bias=True)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(p=0.2)

    self.fc3 = nn.Linear(fc2_out_size, fc3_out_size, bias=True)
  
  def forward(self, x):
    x = self.dropout1(self.relu1(self.fc1(x)))
    x = self.dropout2(self.relu2(self.fc2(x)))
    x = self.fc3(x)

class FlameLayer(nn.Module):
  def __init__(self, **kwargs):
    super(FlameLayer, self).__init__(**kwargs)
  def forward(self, x):
    return x

class Ringnet(nn.Module):
  def __init__(self, resnet, flame):
    super().__init__()
    self.resnet = resnet
    self.flame = flame
  def forward(self, x):
    return x

def train(ringnet):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        res_output = resnet(data)

if __name__ == '__main__':
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
      param.requires_grad = False

    