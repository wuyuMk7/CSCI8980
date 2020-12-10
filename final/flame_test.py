import os

from flame import FLAME
from flame_config import get_config
import numpy as np
import torch
import torch.nn as nn
import trimesh


def batch_orth_proj_idrot(X, camera, name=None):
    """
    X is N x num_points x 3
    camera is N x 3
    same as applying orth_proj_idrot to each N 
    """
    with tf.name_scope(name, "batch_orth_proj_idrot", [X, camera]):
        # TODO check X dim size.
        # tf.Assert(X.shape[2] == 3, [X])

        camera = tf.reshape(camera, [-1, 1, 3], name="cam_adj_shape")

        X_trans = X[:, :, :2] + camera[:, :, 1:]

        shape = tf.shape(X_trans)
        return tf.reshape(
            camera[:, :, 0] * tf.reshape(X_trans, [shape[0], -1]), shape)


def project_points(lmks, camera):
    cam = camera.reshape([-1, 1, 3])
    lmks_trans = lmks[:, :, :2] + cam[:, :, 1:]
    shape = lmks_trans.shape

    lmks_tmp = cam[:, :, 0] * (lmks_trans.reshape([shape[0], -1]))
    return lmks_tmp.reshape(shape)

if __name__ == '__main__':
    config = get_config()
    config.batch_size = 1
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

    cam = torch.rand([1,3]).cuda()
    ret = project_points(landmark, cam)
    print(ret.shape)

    vertices = vertices.detach().cpu().numpy().squeeze()
    faces = flame.faces
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    obj = trimesh.exchange.obj.export_obj(mesh)
    with open('output/test_flame.obj', 'w') as f:
        f.write(obj)