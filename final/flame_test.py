import os

from flame import FLAME
from flame_config import get_config
import numpy as np
import torch
import torch.nn as nn
import trimesh


if __name__ == '__main__':
    config = get_config()
    flame = FLAME(config)

    params_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'test_data', 'params', 'test.npy'
    )
    params = np.load(params_path, allow_pickle=True, encoding='latin1')

    flame.cuda()
    params = params[()]
    print(params['shape'])
    shape_params = torch.tensor(params['shape'].reshape(1,100)).cuda()
    expression_params = torch.tensor(params['expression'].reshape(1,50)).cuda()
    pose_params = torch.tensor(params['pose'].reshape(1,6)).cuda()

    print(shape_params.size())
    # vs, landmarks = flame(shape_params, expression_params, pose_params)
    # print(vs.size(), landmarks.size())

    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
    vertices, landmark = flame(shape_params, expression_params, pose_params) # For RingNet project
    print(vertices.size(), landmark.size())

    vertices = vertices.detach().cpu().numpy().squeeze()
    faces = flame.faces
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    obj = trimesh.exchange.obj.export_obj(mesh)
    with open('output/test_flame.obj', 'w') as f:
        f.write(obj)