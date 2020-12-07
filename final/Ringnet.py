import urllib
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
# from torchsummary import summary
from flame import FLAME
from flame_config import get_config
import skimage.io as io
# from util import renderer as vis_util
from util import image as img_util
from os import listdir, walk
from os.path import isfile, join

dev = "cpu"
if torch.cuda.is_available():  
  dev = "cuda:0"   
device = torch.device(dev)  

res_out_size = 1000
fc1_out_size = 512
fc2_out_size = 512
fc3_out_size = 159

ring_size = 6
ring_size_same_sbj = ring_size - 1
config_img_size = 224

# class Identity(nn.Module):
#   def __init__(self):
#     super(Identity, self).__init__()
  
#   def forward(self, x):
#     return x

# class Regression(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.fc1 = nn.Linear(res_out_size, fc1_out_size, bias=True)
#     self.relu1 = nn.ReLU()
#     self.dropout1 = nn.Dropout(p=0.2)

#     self.fc2 = nn.Linear(fc1_out_size, fc2_out_size, bias=True)
#     self.relu2 = nn.ReLU()
#     self.dropout2 = nn.Dropout(p=0.2)

#     self.fc3 = nn.Linear(fc2_out_size, fc3_out_size, bias=True)
  
#   def forward(self, x):
#     x = self.dropout1(self.relu1(self.fc1(x)))
#     x = self.dropout2(self.relu2(self.fc2(x)))
#     x = self.fc3(x)

# class FlameLayer(nn.Module):
#   def __init__(self, **kwargs):
#   #   super(FlameLayer, self).__init__(**kwargs)
  # def forward(self, x):
  #   return x

class Ringnet(nn.Module):
  def __init__(self, resnet, flame):
    super(Ringnet, self).__init__()
    self.resnet = resnet
    self.fc1 = nn.Linear(res_out_size, fc1_out_size, bias=True)
    self.fc2 = nn.Linear(fc1_out_size, fc2_out_size, bias=True)
    self.fc3 = nn.Linear(fc2_out_size, fc3_out_size, bias=True)
    self.flame = flame
  def forward(self, x):
    x = self.resnet(x)
    print(x.size())
    x = F.dropout(F.relu(self.fc1(x)), p=0.2)
    x = F.dropout(F.relu(self.fc2(x)), p=0.2)
    x = self.fc3(x)
    # shape_params = torch.tensor(params['shape'].reshape(1,100)).cuda()
    # expression_params = torch.tensor(params['expression'].reshape(1,50)).cuda()
    # pose_params = torch.tensor(params['pose'].reshape(1,6)).cuda()
    pose_params = x[:, 3:9] 
    expression_params = x[:, 9:59]
    shape_params = x[:, 59:159]
    print(shape_params.size())
    print(self.flame)
    vertices, landmarks = self.flame(shape_params, expression_params, pose_params)
    return landmarks

# def train(ringnet):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         res_output = resnet(data)

def preprocess_image(img_path):
    img = io.imread(img_path)
    if np.max(img.shape[:2]) != config_img_size:
      print('Resizing so the max image size is %d..' % config_img_size)
      scale = (float(config_img_size) / np.max(img.shape[:2]))
    else:
      scale = 1.0#scaling_factor
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config_img_size)
    # import ipdb; ipdb.set_trace()
    # Normalize image to [-1, 1]
    # plt.imshow(crop/255.0)
    # plt.show()
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

if __name__ == '__main__':
    config = get_config()
    resnet50 = models.resnet50(pretrained=True)
    flameLayer = FLAME(config)
    flameLayer.requires_grad_(False)
    ring = Ringnet(resnet50, flameLayer)
    ring.cuda()

    # get all the training dataset
    pathToTrainingSet = "./training_set/NoW_Dataset/final_release_version/"
    # using a short id list here for debugging convenience, you can create one with 2 id in it 
    subjects_id = open(pathToTrainingSet+'subjects_idst.txt', 'r')
    ids = subjects_id.readlines()

    # cache an image from last person for ring
    prevID = ids[len(ids) - 1]
    for id in ids:
      id = id.rstrip('\n')
      imgPath = pathToTrainingSet + "iphone_pictures/" + id + "/"
      imgDirs = walk(imgPath, topdown=False)
      # image path to six images
      sixRings = []
      for (root, dirs, files) in imgDirs:
        if (root.endswith("TA/")):
          continue
        else:
          for file in files:
            sixRings.append(root + '/' + file)
      
      ringCount = len(sixRings)
      # delete some images if we have more than 5
      if (ringCount > ring_size_same_sbj):
        overSize = ringCount - ring_size_same_sbj
        jump = ringCount // overSize
        for i in range(overSize):
          del sixRings[(overSize-1-i)*jump + jump // 2]
      elif (ringCount < ring_size_same_sbj):
        # we didn't have enough images for a ring
        continue  

      imgs = []
      # get all six images
      for imagePath in sixRings:
        crop, proc_param, img = preprocess_image(imagePath)
        # send into the model
        imgs.append(np.transpose(crop, (2, 0, 1)))

      imgs = np.array(imgs)
      imgs = torch.Tensor(imgs)
      imgs = imgs.cuda()
      vertices, flame_parameters = ring.forward(imgs)

    # use open pose to get their landmarks

    # put them into our model(ringnet)

    # calculate the loss

    # optimze





    