import urllib
from warnings import resetwarnings
from PIL import Image
import numpy as np
from numpy.core.defchararray import count
from numpy.core.fromnumeric import size, transpose
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
import sys

import pyrender
import trimesh
# dir_path = os.getcwd()
# sys.path.append(dir_path + r"\openpose\build\python\openpose\Release")
# os.environ['PATH']  = os.environ['PATH'] + dir_path + r"\openpose\build\x64\Release;" +  dir_path + r"\openpose\build\bin"
# # print(os.environ['PATH'])
# import pyopenpose

# torch.set_printoptions(precision=sys.maxsize)
dev = "cpu"
# if torch.cuda.is_available():  
#   dev = "cuda:0"   
device = torch.device(dev)  

res_out_size = 2048
fc1_out_size = 512
fc2_out_size = 512
fc3_out_size = 159

train_batch_size = 1
learning_rate = 1e-4
epochs = 6
decay = 0.1
shape_loss_eta = 0.5
sc_loss_lambda = 1.0
proj_loss_lambda = 60
feat_loss_shape_lambda = 1e-4
feat_loss_expression_lambda = 1e-4

res_out_size = 2048
fc1_size = 512
fc2_size = 512
regress_out_size = 159
regress_in_size = res_out_size + regress_out_size
regress_iteration_cnt = 8

ring_size = 6
ring_size_same_sbj = ring_size - 1
config_img_size = 224

def project_points(lmks, camera):
  # print(lmks.size())
  # print(camera)
  # cam = camera.reshape([-1, 1, 3])
  # lmks_trans = lmks[:, :, :2] + cam[:, :, 1:]
  # shape = lmks_trans.shape

  # lmks_tmp = cam[:, :, 0] * (lmks_trans.reshape([shape[0], -1]))
  # result = lmks_tmp.reshape(shape)
  # lmks_trans = torch.zeros((lmks.size()[0], lmks.size()[1], lmks.size()[2] - 1))
  temp = camera[:, 0] * lmks[:, :, :2].permute(1, 2, 0)
  temp = temp.permute(0, 2, 1)
  temp += camera[:, 1:]
  result = temp.permute(1, 0, 2)
  return result

'''
Used to turn the translation part of the cam according to the img size
'''
def transform_cam(cam, undo_scale, config_img_size, center):
  temp = cam.permute(1, 0) * undo_scale * config_img_size
  temp[1:, :] += center.permute(1, 0)
  temp = temp.permute(1, 0)
  return temp



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


def preprocess_image(img_path):
    img = io.imread(img_path)
    if np.max(img.shape[:2]) != config_img_size:
      # print('Resizing so the max image size is %d..' % config_img_size)
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
    plt.imshow(crop/255.0)
    plt.show()
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def train(
  dataloader, resnet50, regression, flamelayer, 
  optimizer_reg, optimizer_res, device=None):

  NoWDataLoader = dataloader
  for batch_idx, data_batched in enumerate(NoWDataLoader):
    # print(batch_idx, data_batched['images'].shape)
    cur_batch, cur_batch_shape = data_batched['images'], data_batched['images'].shape
    cur_facepos = data_batched['faceposes']
    cur_img_shapes = data_batched['shapes']
    cur_scales = data_batched['scales']
    cur_centers = data_batched['centers']

    cur_facepos.cuda()
    cur_img_shapes.cuda()
    reshaped_batch = cur_batch.permute(1, 0, 2, 3, 4)
    reshaped_scales = cur_scales.permute(1, 0)
    reshaped_centers = cur_centers.permute(1, 0, 2)
    # reshaped_facepos = cur_facepos.permute(1, 0, 2, 3, 4)

    # Output for each image in the ring
    # regress_outputs, flame_vertices, flame_lmks, flame_cams, flame_proj_lmks = [], [], [], [], []
    regress_outputs, flame_vertices, flame_proj_lmks = [], [], []
    shape_norms, exp_norms = 0.0, 0.0
    for (img_batch, scales_batch, centers_batch ) in zip(reshaped_batch, reshaped_scales, reshaped_centers):
      # cuda!!
      scales_batch = scales_batch.cuda()
      centers_batch = centers_batch.cuda()
      # ResNet50 
      res_output = resnet50(img_batch.float().cuda())

      # Empty estimates as the initial value for concatenation
      regress_estimates = torch.zeros([ res_output.shape[0], regress_out_size ]).cuda()
      # Regression model
      for _ in range(regress_iteration_cnt):
        # Preprocess regression input - concatenation
        regress_input = torch.cat([res_output, regress_estimates], 1)
        regress_estimates = regression(regress_input)
      regress_output = regress_estimates
      regress_outputs.append(regress_output)

      # FLAME model
      cam_params, pose_params = regress_output[0:, 0:3], regress_output[0:, 3:9]
      shape_params, exp_params = regress_output[0:, 9:109], regress_output[0:, 109:159]
      # print(cam_params.shape, pose_params.shape, shape_params.shape, exp_params.shape)
      flame_vert, flame_lmk = flamelayer(shape_params, exp_params, pose_params)
      flame_lmk[:, :, 1] *= -1
      flame_vertices.append(flame_vert)
      # flame_lmks.append(flame_lmk)
      # flame_cams.append(cam_params)
      # print("raw flame landmarks")
      # print(flame_lmk)
      # print("cam ")
      # print(cam_params)
      
      undo_scales_batch = 1. / scales_batch
      cam_params = transform_cam(cam_params, undo_scales_batch, config_img_size, centers_batch)
      flame_proj_lmk = project_points(flame_lmk, cam_params)
      flame_proj_lmks.append(flame_proj_lmk)
      # print(shape_params)
      shape_norms += torch.square(torch.norm(shape_params))
      exp_norms += torch.square(torch.norm(exp_params))
    # print("norms" + str(shape_norms))
    shape_norms = shape_norms / float(len(reshaped_batch))
    exp_norms = exp_norms / float(len(reshaped_batch))
    
    # SC Loss
    diff_idx = len(regress_outputs) - 1
    loss_s = 0.0
    for cur_i in range(diff_idx):
      for next_i in range(diff_idx): 
        if next_i == cur_i: 
          continue
        # cur_same_loss_s = F.mse_loss(regress_outputs[cur_i], regress_outputs[next_i])
        # cur_dif_loss_s = F.mse_loss(regress_outputs[cur_i], regress_outputs[diff_idx])
        cur_same_loss_s = F.mse_loss(regress_outputs[cur_i][3:], regress_outputs[next_i][3:])
        cur_dif_loss_s = F.mse_loss(regress_outputs[cur_i][3:], regress_outputs[diff_idx][3:])
        loss_s += max(0, cur_same_loss_s - cur_dif_loss_s + shape_loss_eta)
    # TODO: Change the scaler!!!! It's wrong!!! - Can add average=False to loss !!!
    loss_sc = loss_s / (len(reshaped_batch) * ring_size)
    # loss_sc = loss_s / ring_size

    # Proj Loss
    loss_proj = 0.0
    # first get the landmarks
    reshaped_faceposes = cur_facepos.permute(1, 0, 2, 3, 4)
    reshaped_img_shapes = cur_img_shapes.permute(1, 0, 2, 3)
    for img_idx in range(len(reshaped_batch)):
      flame_proj_lmk, reshaped_facepos, img_shapes = flame_proj_lmks[img_idx].cuda(), reshaped_faceposes[img_idx].cuda(), reshaped_img_shapes[img_idx].cuda()
      # flame_proj_lmk, reshaped_facepos, img_shapes = flame_proj_lmks[img_idx], reshaped_faceposes[img_idx], reshaped_img_shapes[img_idx]
      ground_truth_weights = ((reshaped_facepos[:,:,:,2] > 0.41).float()).reshape(-1, 68)
      # print(ground_truth_weights)
      ground_truth_lmk = (reshaped_facepos[:,:,:,:2]).reshape(-1, 68, 2)
      #print(ground_truth_weights)
      ground_truth_weights = ground_truth_weights.unsqueeze(2)
      ground_truth_weights = torch.cat([ground_truth_weights, ground_truth_weights], 2).cuda()

      # 
      scale_cpu = reshaped_scales[img_idx].cuda()
      input = (ground_truth_weights * flame_proj_lmk).permute(1,2,0) * scale_cpu
      ground_truth = (ground_truth_weights * ground_truth_lmk).permute(1,2,0) * scale_cpu
      input = input.permute(2,0,1)
      ground_truth = ground_truth.permute(2,0,1)
      # print(input)
      # print(ground_truth)
      loss_p = F.l1_loss(
        # size_average=False,
        input = input,
        target = ground_truth
      )
      # print(scale_cpu)
      # print(config_img_size)
      # print(loss_p)
      loss_proj += loss_p / (config_img_size)
      # print(loss_proj)
    loss_proj = loss_proj / (len(reshaped_batch))

    # Total Loss
    sc_part = sc_loss_lambda * loss_sc 
    proj_part = proj_loss_lambda * loss_proj
    # loss_tot = 0.0
    shape_part = feat_loss_shape_lambda * shape_norms
    exp_part = feat_loss_expression_lambda * exp_norms
    # loss_tot = sc_part + proj_part + shape_part + exp_part
    loss_tot = proj_part
    print("sc: " + str(sc_part) + " proj: " + str(proj_part) + " shape: " + str(shape_part) + " exp: " + str(exp_part))
    print(batch_idx, loss_tot)
    optimizer_reg.zero_grad()
    optimizer_res.zero_grad()
    loss_tot.backward()
    optimizer_reg.step()
    optimizer_res.step()



def evaluate(resnet50, regression, flamelayer, dataloader):

  NoWDataLoader = dataloader
  for batch_idx, data_batched in enumerate(NoWDataLoader):
    # print(batch_idx, data_batched['images'].shape)
    cur_batch, cur_batch_shape = data_batched['images'], data_batched['images'].shape
    cur_facepos = data_batched['faceposes']
    cur_facepos.cuda()
    reshaped_batch = cur_batch.permute(1, 0, 2, 3, 4)
    # reshaped_facepos = cur_facepos.permute(1, 0, 2, 3, 4)
    # print(cur_facepos.size())
    # print(reshaped_facepos.size())

    # Output for each image in the ring
    # regress_outputs, flame_vertices, flame_lmks, flame_cams, flame_proj_lmks = [], [], [], [], []
    regress_outputs, flame_vertices, flame_proj_lmks = [], [], []
    shape_norms, exp_norms = 0.0, 0.0
    # print(reshaped_batch.size())
    for img_batch in reshaped_batch:
      # ResNet50 
      # res_output = resnet50(img_batch.float())
      res_output = resnet50(img_batch.float().cuda())

      # Empty estimates as the initial value for concatenation
      regress_estimates = torch.zeros([ res_output.shape[0], regress_out_size ]).cuda()
      # Regression model
      for _ in range(regress_iteration_cnt):
        # Preprocess regression input - concatenation
        regress_input = torch.cat([res_output, regress_estimates], 1)
        regress_estimates = regression(regress_input)
      regress_output = regress_estimates
      regress_outputs.append(regress_output)

      # FLAME model
      cam_params, pose_params = regress_output[0:, 0:3], regress_output[0:, 3:9]
      shape_params, exp_params = regress_output[0:, 9:109], regress_output[0:, 109:159]
      # print(cam_params.shape, pose_params.shape, shape_params.shape, exp_params.shape)
      flame_vert, flame_lmk = flamelayer(shape_params, exp_params, pose_params)
      flame_vertices.append(flame_vert)
      # flame_lmks.append(flame_lmk)
      # flame_cams.append(cam_params)
      # print("raw flame landmarks")
      # print(flame_lmk)
      # print("cam ")
      # print(cam_params)
      flame_proj_lmk = project_points(flame_lmk, cam_params)

      faces = flamelayer.faces
      vertices = flame_vert.cpu().detach().numpy()[0]
      joints = flame_lmk.cpu().detach().numpy()[0]
      # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

      # print("All info about faces: 1. length 2. itself")
      # print(len(faces))
      # print(faces)
      tri_mesh = trimesh.Trimesh(vertices, faces)
      mesh = pyrender.Mesh.from_trimesh(tri_mesh)
      scene = pyrender.Scene()
      scene.add(mesh)
      sm = trimesh.creation.uv_sphere(radius=0.005)
      sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
      tfs = np.tile(np.eye(4), (len(joints), 1, 1))
      tfs[:, :3, 3] = joints
      joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
      scene.add(joints_pcl)
      scale = cam_params[0, 0]
      oc = pyrender.OrthographicCamera(scale, scale)
      pyrender.Viewer(scene, use_raymond_lighting=True)



      flame_proj_lmks.append(flame_proj_lmk)
      shape_norms += torch.norm(shape_params)
      exp_norms += torch.norm(exp_params)
    

def weight_init(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)

if __name__ == '__main__':
  need_evaluate = False
  composed_transforms = transforms.Compose([ ScaleAndCrop(config_img_size), ToTensor() ])
  dataset = NoWDataset(
    dataset_path = os.path.join('.', 'training_set', 'NoW_Dataset', 'final_release_version'), 
    data_folder = 'iphone_pictures',
    facepos_folder= 'openpose',
    id_txt = 'subjects_idst.txt',
    R = 6,
    transform = composed_transforms
  )

  if need_evaluate:
    resnet50 = torch.load("./resnet50.pkl")
  else:
    resnet50 = models.resnet50(pretrained=True)
  resnet50.cuda()
  resnet50.fc = Identity()
  if need_evaluate:
    regression = torch.load("./model.pkl")
  else:
    regression = Regression()
  regression.cuda()
  config = get_config()
  flamelayer = FLAME(config)
  flamelayer.requires_grad_ = False
  flamelayer.cuda()

  # ringnet = SingleRingnet(resnet50, regression, flamelayer)
  # ringnet.apply(weight_init)
  # ringnet.cuda()

  optimizer_reg = torch.optim.Adam(regression.parameters(), lr=learning_rate)
  optimizer_res = torch.optim.Adam(resnet50.parameters(), lr=learning_rate)
  scheduler_reg = torch.optim.lr_scheduler.StepLR(optimizer_reg, step_size=5)
  scheduler_res = torch.optim.lr_scheduler.StepLR(optimizer_res, step_size=5)

  # img = dataset[0]['images'][0].permute(1,2,0).numpy()
  # img = dataset[0]['images'][0].permute(1,2,0)
  # print(img)
  # feature = resnet50(dataset[0]['images'][0].reshape(-1,3,224,224).float())
  # print(feature.shape)

  NoWDataLoader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
  
  if need_evaluate:
    evaluate(resnet50, regression, flamelayer, NoWDataLoader)
  else:
    for epoch_idx in range(epochs):
      print("Epoch: {}".format(epoch_idx))
      train(NoWDataLoader, resnet50, regression, flamelayer, optimizer_reg, optimizer_res)
    torch.save(regression, "./model.pkl")
    torch.save(resnet50, "./resnet50.pkl")
    # print(cur_batch_shape, reshaped_batch[0].shape)
    # plt.imshow(reshaped_batch[0][0].permute(1,2,0).numpy())
    # plt.show()
    # break
    
    #ioutput = resnet50(data)

  # all_img_paths = load_NoWdataset()
  #print(all_img_paths)
  # ring.cuda()

  # shorter_id.txt
  # Load data


    