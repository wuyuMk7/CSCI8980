from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from absl import flags
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
# import tensorflow as tf
# from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model

# from util import renderer as vis_util
from util import image as img_util

from flame import FLAME
from flame_config import get_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

import MyRingnet

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
    # init.normal_(self.fc1, 0, 1)
    # init.normal_(self.fc2, 0, 1)
    # init.normal_(self.fc3, 0, 1)
  
  def forward(self, x):
    x = self.dropout1(self.relu1(self.fc1(x)))
    x = self.dropout2(self.relu2(self.fc2(x)))
    # x = self.relu1(self.fc1(x))
    # x = self.relu2(self.fc2(x))
    x = self.fc3(x)
    return x

# if __name__ == '__main__':
    config = get_config()
    template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')
    renderer = vis_util.SMPLRenderer(faces=template_mesh.f)

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    if not os.path.exists(config.out_folder + '/images'):
        os.mkdir(config.out_folder + '/images')

    main(config, template_mesh)

config_img_size = 244

if __name__ == '__main__':
    # read images and scale
    input_img_path = "./training_set/NoW_Dataset/final_release_version/iphone_pictures/FaMoS_180424_03335_TA/multiview_neutral/IMG_0101.jpg"
    openpose = np.load(input_img_path.replace("iphone_pictures", "openpose").replace("jpg", "npy"), allow_pickle=True, encoding='latin1')
    img = io.imread(input_img_path)
    if np.max(img.shape[:2]) != config_img_size:
        # print('Resizing so the max image size is %d..' % self.config_img_size)
        scale = (float(config_img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.0#scaling_factor
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(
        img, scale, center, config_img_size)

    crop = torch.tensor(crop)
    crop = crop.permute(2, 0, 1)
    crop = crop[None, :, :, :].float().cuda()
    # print(crop)
    # build model
    resnet50 = torch.load("./resnet50.pkl")
    resnet50.cuda()
    resnet50.fc = Identity()
    # print(resnet50)
    regression = torch.load("./model.pkl")
    regression.cuda()
    config = get_config()
    config.batch_size = 1
    flamelayer = FLAME(config)
    flamelayer.requires_grad_ = False
    flamelayer.cuda()

    # run the model
    res_output = resnet50(crop)
    # Empty estimates as the initial value for concatenation
    regress_estimates = torch.zeros([ res_output.shape[0], MyRingnet.regress_out_size ]).cuda()
    # Regression model
    for _ in range(MyRingnet.regress_iteration_cnt):
      # Preprocess regression input - concatenation
      regress_input = torch.cat([res_output, regress_estimates], 1)
      regress_estimates = regression(regress_input)
    regress_output = regress_estimates
    # FLAME model
    cam_params, pose_params = regress_output[0:, 0:3], regress_output[0:, 3:9]
    shape_params, exp_params = regress_output[0:, 9:109], regress_output[0:, 109:159]
    # pose_params[0,2] = 3.14/5
    flame_vert, flame_lmk = flamelayer(shape_params, exp_params, pose_params)
    flame_lmk[:, :, 1] *= -1

    # cam_params[:, 0] = 2
    # cam_params[:, 1] = 0.2
    print(cam_params)
    center = torch.tensor(center.copy()).cuda()
    new_cam = MyRingnet.transform_cam(cam_params, 1. / scale, config_img_size, center[None, :])
    projected_lmks = MyRingnet.project_points(flame_lmk, new_cam)
    print(projected_lmks)
    print(openpose)
    plt.figure
    plt.imshow(img)
    count = 0
    cpu_lmks = projected_lmks.cpu()
    for i in cpu_lmks[0]:
        x = i[0].int()
        y = i[1].int()
        plt.annotate(str(count), xy=(x, y))
        plt.scatter(x, y, s=50, c='red', marker='o')
        count = count + 1
    count = 0
    for i in openpose[0]:
        x = i[0]
        y = i[1]
        plt.annotate(str(count), xy=(x, y))
        plt.scatter(x, y, s=50, c='blue', marker='o')
        count = count + 1
    plt.show()