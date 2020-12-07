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
import random, os

import matplotlib.pyplot as plt
from dataset import NoWDataset, ScaleAndCrop, ToTensor
from torch.utils.data import DataLoader

dev = "cpu"
if torch.cuda.is_available():  
  dev = "cuda:0"   
device = torch.device(dev)  

res_out_size = 2048
fc1_out_size = 512
fc2_out_size = 512
fc3_out_size = 159

res_out_size = 2048
fc1_size = 512
fc2_size = 512
regress_out_size = 159
regress_in_size = res_out_size + regress_out_size
regress_iteration_cnt = 8

ring_size = 6
ring_size_same_sbj = ring_size - 1
config_img_size = 224

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  
  def forward(self, x):
    return x

# Input size: 2048 + 159, fc1_size: 512, fc2_size: 512, out_size: 159
class Regression(nn.Module):
  def __init__(
    self, input_size = 2048+159, fc1_size = 512, 
    fc2_size = 512, out_size = 159, iter = 8):
    super().__init__()
    self.fc1 = nn.Linear(input_size, fc1_size, bias=True)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(p=0.2)

    self.fc2 = nn.Linear(fc1_size, fc2_size, bias = True)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(p=0.2)

    self.fc3 = nn.Linear(fc2_size, out_size, bias=True)
  
  def forward(self, x):
    x = self.dropout1(self.relu1(self.fc1(x)))
    x = self.dropout2(self.relu2(self.fc2(x)))
    x = self.fc3(x)
    return x

# class Ringnet(nn.Module):
#   def __init__(self, resnet, flame):
#     super(Ringnet, self).__init__()
#     self.resnet = resnet
#     self.fc1 = nn.Linear(res_out_size, fc1_out_size, bias=True)
#     self.fc2 = nn.Linear(fc1_out_size, fc2_out_size, bias=True)
#     self.fc3 = nn.Linear(fc2_out_size, fc3_out_size, bias=True)
#     self.flame = flame
#   def forward(self, x):
#     x = self.resnet(x)
#     print(x.size())
#     x = F.dropout(F.relu(self.fc1(x)), p=0.2)
#     x = F.dropout(F.relu(self.fc2(x)), p=0.2)
#     x = self.fc3(x)
#     # shape_params = torch.tensor(params['shape'].reshape(1,100)).cuda()
#     # expression_params = torch.tensor(params['expression'].reshape(1,50)).cuda()
#     # pose_params = torch.tensor(params['pose'].reshape(1,6)).cuda()
#     pose_params = x[:, 3:9] 
#     expression_params = x[:, 6:59]
#     shape_params = x[:, 59:159]
#     print(shape_params.size())
#     print(self.flame)
#     vertices, landmarks = self.flame(shape_params, expression_params, pose_params)
#     return landmarks

class SingleRingnet(nn.Module):
  def __init__(self, resnet, regression, flame):
    super(SingleRingnet, self).__init__()
    self.resnet = resnet
    self.regression = regression
    self.flame = flame

    # for param in self.resnet.parameters():
    #   param.requires_grad = False
    # for param in self.flame.parameters():
    #   param.requires_grad = False

  def forward(self, x):
    with torch.no_grad:
      x = self.resnet(x)
    x = self.regression(x)
    with torch.no_grad:
      x = self.flame(x)
    return x

# class Ringnet(nn.Module):
#   def __init__(self, )

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

# if __name__ == '__main__':
#     config = get_config()
#     resnet50 = models.resnet50(pretrained=True)
#     flameLayer = FLAME(config)
#     flameLayer.requires_grad_(False)
#     ring = Ringnet(resnet50, flameLayer)
#     ring.cuda()

#     # get all the training dataset
#     pathToTrainingSet = "./training_set/NoW_Dataset/final_release_version/"
#     # using a short id list here for debugging convenience, you can create one with 2 id in it 
#     subjects_id = open(pathToTrainingSet+'subjects_idst.txt', 'r')
#     ids = subjects_id.readlines()

#     # cache an image from last person for ring
#     prevID = ids[len(ids) - 1]
#     for id in ids:
#       id = id.rstrip('\n')
#       imgPath = pathToTrainingSet + "iphone_pictures/" + id + "/"
#       imgDirs = walk(imgPath, topdown=False)
#       # image path to six images
#       sixRings = []
#       for (root, dirs, files) in imgDirs:
#         if (root.endswith("TA/")):
#           continue
#         else:
#           for file in files:
#             sixRings.append(root + '/' + file)
      
#       ringCount = len(sixRings)
#       # delete some images if we have more than 6
#       if (ringCount > ring_size_same_sbj):
#         overSize = ringCount - ring_size_same_sbj
#         jump = ringCount // overSize
#         for i in range(overSize-1):
#           del sixRings[(overSize-1-i)*jump + jump // 2]
#       elif (ringCount < ring_size_same_sbj):
#         # we didn't have enough images for a ring
#         continue  

#       imgs = []
#       # get all six images
#       for imagePath in sixRings:
#         crop, proc_param, img = preprocess_image(imagePath)
#         # send into the model
#         imgs.append(np.transpose(crop, (2, 0, 1)))

#       imgs = np.array(imgs)
#       imgs = torch.Tensor(imgs)
#       imgs = imgs.cuda()
#       vertices, flame_parameters = ring.forward(imgs)

#     # use open pose to get their landmarks

#     # put them into our model(ringnet)

#     # calculate the loss

#     # optimze

# def load_NoWdataset(
#   dataset_path = os.path.join('.', 'training_set', 'NoW_Dataset', 'final_release_version'), 
#   id_txt = 'subjects_idst.txt',
#   data_folder = 'iphone_pictures',
#   R = 6):
#       # get all the training dataset
#     pathToTrainingSet = "./training_set/NoW_Dataset/final_release_version/"
#     # using a short id list here for debugging convenience, you can create one with 2 id in it 
#     subjects_id = open(pathToTrainingSet+'subjects_idst.txt', 'r')
#     ids = subjects_id.readlines()

#     id_file = os.path.join(dataset_path, id_txt)
#     subject_ids = []
#     with open(id_file, 'r') as f:
#       subject_ids = f.read().split()
#     subject_paths = [ 
#       os.path.join(dataset_path, data_folder, subject_id) 
#       for subject_id in subject_ids
#     ]

#     all_img_paths = [ 
#       [ os.path.join(subject_path, sub_img_dir) for sub_img_dir in os.listdir(subject_path) ] 
#       for subject_path in subject_paths 
#     ]
#     all_imgs = [
#       [ os.path.join(img_path, img) for img_path in img_paths for img in os.listdir(img_path)]
#       for img_paths in all_img_paths
#     ]      

#     # selected_img_paths = [ [ [imgs from the cur person], [img from another person] ] ]
#     selected_imgs_paths = []
#     random_sample_range = (0, len(subject_paths)-1)
#     for cur_i in range(len(subject_paths)):
#       next_i = cur_i
#       while len(subject_paths) > 1 and next_i == cur_i:
#         next_i = random.randint(random_sample_range[0], random_sample_range[1])
#       cur_R = R if len(all_imgs[cur_i]) > R else len(all_imgs[cur_i])
#       same_img_paths = random.sample(all_imgs[cur_i], cur_R)
#       dif_img_path = all_imgs[next_i][random.randint(0, len(all_imgs[next_i])-1)]
#       selected_imgs_paths.append([ same_img_paths, [ dif_img_path ] ])  
  
#     return selected_imgs_paths

#     #   imgs = []
#     #   # get all six images
#     #   for imagePath in sixRings:
#     #     crop, proc_param, img = preprocess_image(imagePath)
#     #     # send into the model
#     #     imgs.append(np.transpose(crop, (2, 0, 1)))

#     #   imgs = np.array(imgs)
#     #   imgs = torch.Tensor(imgs)
#     #   imgs = imgs.cuda()
#     #   vertices, flame_parameters = ring.forward(imgs)



if __name__ == '__main__':
  composed_transforms = transforms.Compose([ ScaleAndCrop(config_img_size), ToTensor() ])
  dataset = NoWDataset(
    dataset_path = os.path.join('.', 'training_set', 'NoW_Dataset', 'final_release_version'), 
    data_folder = 'iphone_pictures',
    id_txt = 'subjects_idst.txt',
    R = 6,
    transform = composed_transforms
  )

  resnet50 = models.resnet50(pretrained=True)
  resnet50.fc = Identity()
  regression = Regression()
  config = get_config()
  flamelayer = FLAME(config)

  ringnet = SingleRingnet(resnet50, regression, flamelayer)
  # print(ringnet)

  # img = dataset[0]['images'][0].permute(1,2,0).numpy()
  # img = dataset[0]['images'][0].permute(1,2,0)
  # print(img)
  # feature = resnet50(dataset[0]['images'][0].reshape(-1,3,224,224).float())
  # print(feature.shape)

  NoWDataLoader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=1)
  for batch_idx, data_batched in enumerate(NoWDataLoader):
    cur_batch, cur_batch_shape = data_batched['images'], data_batched['images'].shape
    reshaped_batch = cur_batch.permute(1, 0, 2, 3, 4)

    # Output for each image in the ring
    flame_outputs = []
    for img_batch in reshaped_batch:
      # ResNet50 
      res_output = resnet50(img_batch.float())

      # Empty estimates as the initial value for concatenation
      regress_estimates = torch.zeros([ res_output.shape[0], regress_out_size ])
      # Regression model
      for _ in range(regress_iteration_cnt):
        # Preprocess regression input - concatenation
        regress_input = torch.cat([res_output, regress_estimates], 1)
        regress_estimates = regression(regress_input)
      regress_output = regress_estimates

      # FLAME model
      cam_params, pose_params = regress_output[0:, 0:3], regress_output[0:, 3:9]
      shape_params, exp_params = regress_output[0:, 9:109], regress_output[0:, 109:159]
      # print(cam_params.shape, pose_params.shape, shape_params.shape, exp_params.shape)
      flame_output = flamelayer(shape_params, exp_params, pose_params)
      print(flame_output)

      exit(0)
    
    # SC Loss
    dif_idx = len(regress_outputs_imgs) - 1
    loss_sc = None 
    for sc_loss_i in range(dif_idx - 1):
      pass
    exit(0)
      

    # Proj Loss
    loss_proj = 0.0

    # Total Loss
    loss_tot = 0.0

    # print(cur_batch_shape, reshaped_batch[0].shape)
    # plt.imshow(reshaped_batch[0][0].permute(1,2,0).numpy())
    # plt.show()
    # break
    
    #output = resnet50(data)

  # all_img_paths = load_NoWdataset()
  #print(all_img_paths)
  # ring.cuda()

  # shorter_id.txt
  # Load data


    