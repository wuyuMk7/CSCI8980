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

from util import renderer as vis_util
from util import image as img_util
from util.project_on_mesh import compute_texture_map

from flame import FLAME
from flame_config import get_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import transforms, modelsV
def visualize(img, proc_param, verts, cam, img_name='test_image'):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted = vis_util.get_original(
        proc_param, verts, cam, img_size=img.shape[:2])

    # Render results
    rend_img_overlay = renderer(
        vert_shifted*1.0, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted*1.0, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 30, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show(block=False)
    fig.savefig(img_name + '.png')
    # import ipdb
    # ipdb.set_trace()


def create_texture(img, proc_param, verts, faces, cam, texture_data):
    cam_for_render, vert_shifted = vis_util.get_original(proc_param, verts, cam, img_size=img.shape[:2])

    texture_map = compute_texture_map(img, vert_shifted, faces, cam_for_render, texture_data)
    return texture_map


def preprocess_image(img_path):
    img = io.imread(img_path)
    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.0#scaling_factor
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # import ipdb; ipdb.set_trace()
    # Normalize image to [-1, 1]
    # plt.imshow(crop/255.0)
    # plt.show()
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(config, template_mesh):
    # sess = tf.Session()
    # model = RingNet_inference(config, sess=sess)

    # build up the model

    model = 




    input_img, proc_param, img = preprocess_image(config.img_path)
    vertices, flame_parameters = model.predict(np.expand_dims(input_img, axis=0), get_parameters=True)
    cams = flame_parameters[0][:3]
    visualize(img, proc_param, vertices[0], cams, img_name=config.out_folder + '/images/' + config.img_path.split('/')[-1][:-4])

    if config.save_obj_file:
        if not os.path.exists(config.out_folder + '/mesh'):
            os.mkdir(config.out_folder + '/mesh')
        mesh = Mesh(v=vertices[0], f=template_mesh.f)
        mesh.write_obj(config.out_folder + '/mesh/' + config.img_path.split('/')[-1][:-4] + '.obj')

    if config.save_flame_parameters:
        if not os.path.exists(config.out_folder + '/params'):
            os.mkdir(config.out_folder + '/params')
        flame_parameters_ = {'cam':  flame_parameters[0][:3], 'pose': flame_parameters[0][3:3+config.pose_params], 'shape': flame_parameters[0][3+config.pose_params:3+config.pose_params+config.shape_params],
         'expression': flame_parameters[0][3+config.pose_params+config.shape_params:]}
        np.save(config.out_folder + '/params/' + config.img_path.split('/')[-1][:-4] + '.npy', flame_parameters_)

    if config.neutralize_expression:
        from util.using_flame_parameters import make_prdicted_mesh_neutral
        if not os.path.exists(config.out_folder + '/neutral_mesh'):
            os.mkdir(config.out_folder + '/neutral_mesh')
        neutral_mesh = make_prdicted_mesh_neutral(config.out_folder + '/params/' + config.img_path.split('/')[-1][:-4] + '.npy', config.flame_model_path)
        neutral_mesh.write_obj(config.out_folder + '/neutral_mesh/' + config.img_path.split('/')[-1][:-4] + '.obj')

    if config.save_texture:
        if not os.path.exists(config.flame_texture_data_path):
            print('FLAME texture data not found')
            return
        texture_data = np.load(config.flame_texture_data_path, allow_pickle=True)[()]
        texture = create_texture(img, proc_param, vertices[0], template_mesh.f, cams, texture_data)

        if not os.path.exists(config.out_folder + '/texture'):
            os.mkdir(config.out_folder + '/texture')

        cv2.imwrite(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.png', texture[:,:,::-1])
        mesh = Mesh(v=vertices[0], f=template_mesh.f)
        mesh.vt = texture_data['vt']
        mesh.ft = texture_data['ft']
        mesh.set_texture_image(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.png')
        mesh.write_obj(config.out_folder + '/texture/' + config.img_path.split('/')[-1][:-4] + '.obj')


# if __name__ == '__main__':
    config = get_config()
    template_mesh = Mesh(filename='./flame_model/FLAME_sample.ply')
    renderer = vis_util.SMPLRenderer(faces=template_mesh.f)

    if not os.path.exists(config.out_folder):
        os.makedirs(config.out_folder)

    if not os.path.exists(config.out_folder + '/images'):
        os.mkdir(config.out_folder + '/images')

    main(config, template_mesh)


if __name__ == '__main__':
  need_evaluate = True

  resnet50 = torch.load("./resnet50.pkl")
  resnet50.cuda()
  resnet50.fc = Identity()
  regression = torch.load("./model.pkl")
  regression.cuda()
  config = get_config()
  config.batch_size = 1
  flamelayer = FLAME(config)
  flamelayer.requires_grad_ = False
  flamelayer.cuda()