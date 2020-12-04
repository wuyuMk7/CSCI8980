import os
import numpy as np
import chumpy as ch
from smpl_webuser.serialization import load_model
from smpl_webuser.verts import verts_decorated
#from psbody.mesh import Mesh
import trimesh

def make_prdicted_mesh_neutral(predicted_params_path, flame_model_path):
    params = np.load(predicted_params_path, allow_pickle=True, encoding='latin1')
    #print(params)
    params = params[()]
    pose = np.zeros(15)
    #expression = np.zeros(100)
    shape = np.hstack((params['shape'], np.zeros(300-params['shape'].shape[0])))
    #pose = np.hstack((params['pose'], np.zeros(15-params['pose'].shape[0])))
    expression = np.hstack((params['expression'], np.zeros(100-params['expression'].shape[0])))
    flame_genral_model = load_model(flame_model_path)
    generated_neutral_mesh = verts_decorated(
        #ch.array([0.0,0.0,0.0]),
        ch.array(params['cam']),
        ch.array(pose),
        ch.array(flame_genral_model.r),
        flame_genral_model.J_regressor,
        ch.array(flame_genral_model.weights),
        flame_genral_model.kintree_table,
        flame_genral_model.bs_style,
        flame_genral_model.f,
        bs_type=flame_genral_model.bs_type,
        posedirs=ch.array(flame_genral_model.posedirs),
        betas=ch.array(np.hstack((shape,expression))),#betas=ch.array(np.concatenate((theta[0,75:85], np.zeros(390)))), #
        shapedirs=ch.array(flame_genral_model.shapedirs),
        want_Jtr=True
    )
    # neutral_mesh = Mesh(v=generated_neutral_mesh.r, f=generated_neutral_mesh.f)
    neutral_mesh = trimesh.Trimesh(vertices=generated_neutral_mesh.r, faces=generated_neutral_mesh.f)
    return neutral_mesh

if __name__ == '__main__':
    params_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'test_data', 'params', '000001.npy'
    )
    model_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'flame_model', 'generic_model.pkl'
    )
    mesh = make_prdicted_mesh_neutral(params_path, model_path)
    obj = trimesh.exchange.obj.export_obj(mesh)
    with open('output/test_neutral.obj', 'w') as f:
        f.write(obj)
