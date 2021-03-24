import open3d as o3d
import torch
import math
import torch.optim as optim
from tqdm import tqdm
import smplx
from human_body_prior.tools.model_loader import load_vposer
import argparse
import sys, os, glob
import numpy as np
from utils import *
from utils_read_data import *
import pdb


## figure out body model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vposer_model, _ = load_vposer('/home/yzhang/body_models/VPoser/vposer_v1_0/', vp_model='snapshot')
vposer_model = vposer_model.to(device)

smplx_model = smplx.create('/home/yzhang/body_models/VPoser/',
                               model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=50
                               ).to(device)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default='optimize_results', type=str)
    parser.add_argument("--dataset", default='replica', type=str)
    parser.add_argument("--scene_name", default=None)


    args = parser.parse_args()
    scenename = args.scene_name if args.scene_name is not None else '*'
    result_folder_per_scene = glob.glob(os.path.join(args.result_path,
                                                    args.dataset,
                                                    scenename))

    for folder in result_folder_per_scene:
        ## load results of PLACE
        body_params_75 = torch.FloatTensor(np.load(folder+'/body_params_opt_list_s2.npy')).to(device)
        rot_angles = np.load(folder+'/rot_angle_list_1.npy')
        shifts = np.load(folder+'/shift_list.npy')
        # pdb.set_trace()
        ## convert smplx parameters
        body_params_72 = convert_to_3D_rot(body_params_75)  # tensor, [bs=1, 72]
        body_pose = vposer_model.decode(body_params_72[:, 16:48], output_type='aa').view(-1,63)  # tensor, [bs=1, 63]
        body_verts, body_param_dict = gen_body_mesh(body_params_72, body_pose, smplx_model)


        ## first we visualize the place results for safety.
        body_mesh_list = []
        for j in range(10):
            body_verts_one = body_verts.detach().cpu().numpy()[j]
            # transfrom the body verts to the PROX world coordinate system
            ####----------TODO change the transform to transformation matrix, and then update the SMPLX global params in the world coordinate
            body_verts_opt_prox_s2 = np.zeros(body_verts_one.shape)  # [10475, 3]
            temp = body_verts_one - shifts[j]
            body_verts_opt_prox_s2[:, 0] = temp[:, 0] * math.cos(-rot_angles[j]) - temp[:, 1] * math.sin(-rot_angles[j])
            body_verts_opt_prox_s2[:, 1] = temp[:, 0] * math.sin(-rot_angles[j]) + temp[:, 1] * math.cos(-rot_angles[j])
            body_verts_opt_prox_s2[:, 2] = temp[:, 2]

            body_mesh_opt_s2 = o3d.geometry.TriangleMesh()
            body_mesh_opt_s2.vertices = o3d.utility.Vector3dVector(body_verts_opt_prox_s2)
            body_mesh_opt_s2.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
            body_mesh_opt_s2.compute_vertex_normals()
            body_mesh_list.append(body_mesh_opt_s2)
        scene_name = os.path.basename(folder)
        scene_mesh = o3d.io.read_point_cloud(os.path.join(os.path.join('/mnt/hdd/datasets/PlaceInReplica/replica_v1/', scene_name), 'mesh.ply'))
        o3d.visualization.draw_geometries([scene_mesh]+body_mesh_list)




        ## resolve global transformations










