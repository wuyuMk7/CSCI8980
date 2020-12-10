import cv2
import argparse
import time

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
from os.path import isdir, isfile, join
import random, os
from sys import platform

import matplotlib.pyplot as plt
from dataset import NoWDataset, ScaleAndCrop, ToTensor
from torch.utils.data import DataLoader
import sys
dir_path = os.getcwd()
sys.path.append(dir_path + r"\openpose\build\python\openpose\Release")
os.environ['PATH']  = os.environ['PATH'] + dir_path + r"\openpose\build\x64\Release;" +  dir_path + r"\openpose\build\bin"
# print(os.environ['PATH'])
import pyopenpose as op

config_img_size = 224

def preprocess_image(img_path):
    img = io.imread(img_path)
    if np.max(img.shape[:2]) != config_img_size:
    #   print('Resizing so the max image size is %d..' % config_img_size)
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

# get all the training dataset
pathToTrainingSet = "./training_set/NoW_Dataset/final_release_version/"
# using a short id list here for debugging convenience, you can create one with 2 id in it 
subjects_id = open(pathToTrainingSet+'subjects_id.txt', 'r')
ids = subjects_id.readlines()

count = 0
for id in ids:
    count = count + 1
    if count < 99:
        continue
    id = id.rstrip('\n')
    imgPath = pathToTrainingSet + "iphone_pictures/" + id + "/"
    # facePosPath = pathToTrainingSet + "detected_face/" + id + "/"
    # saveLmPath = pathToTrainingSet + "openpose/" + id + "/"
    imgDirs = walk(imgPath, topdown=False)
    for  (root, dirs, files) in imgDirs:
        if (root.endswith("TA/")):
            continue
        else:
            for file in files:
                imgDir = root + '/' +  file
                # load the face positions from npy
                facepos = np.load(imgDir.replace('iphone_pictures', 'detected_face').replace("jpg", "npy"), allow_pickle=True, encoding='latin1')
                params = dict()
                params["model_folder"] = "./openpose/models/"
                params["face"] = True
                params["face_detector"] = 2
                params["body"] = 0 

                # Starting OpenPose
                opWrapper = op.WrapperPython()
                opWrapper.configure(params)
                opWrapper.start()

                # Read image and face rectangle locations
                imageToProcess = cv2.imread(imgDir)
                # print(facepos)
                # facepos.item()['top'], facepos.item()['bottom'], facepos.item()['left'], facepos.item()['right']
                left = facepos.item()['left']
                right = facepos.item()['right']
                top = facepos.item()['top']
                bottom = facepos.item()['bottom']
                lrdiff = right - left
                btdiff = bottom - top
                len = 0
                x = left
                y = top
                if lrdiff < btdiff:
                    len = lrdiff
                    y = y + (btdiff - lrdiff) / 2
                else:
                    len = btdiff
                    x = x + (lrdiff - btdiff) / 2

                faceRectangles = [
                    op.Rectangle(x, y, len, len),
                ]

                # Create new datum
                datum = op.Datum()
                datum.cvInputData = imageToProcess
                datum.faceRectangles = faceRectangles

                # Process and display image
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                # print("Face keypoints: \n" + str(datum.faceKeypoints))

                # save openpose data
                # rootForFacepos = root.replace('iphone_pictures', 'openpose')
                # if not isdir(rootForFacepos):
                #     os.makedirs(rootForFacepos)
                # fileForFacepos = file.replace('jpg', 'npy')
                # fileSave = rootForFacepos + '/' + fileForFacepos
                # np.save(fileSave, datum.faceKeypoints)
                imS = cv2.resize(datum.cvOutputData, (540, 960)) 
                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", imS)
                cv2.waitKey(0)

