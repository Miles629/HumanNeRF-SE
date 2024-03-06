import os
import sys
import os.path as osp
import numpy as np
import argparse
import json
import cv2
import body_model
import glob
from smpl_numpy import SMPL
import pickle
from  tqdm  import tqdm

try:
    import smplx
    import torch
except ImportError:
    print('smplx and torch are needed for visualization of SMPL vertices.')

def transform_points(points, R, T):
    """ Transform 3D points from world coordinate system to camera coordinate system
    Args:
        points (np.ndarray): 3D points in world coordinate system of shape (N, 3).
        R (np.ndarray): world2cam rotation matrix of shape (3, 3).
        T (np.ndarray): world2cam translation vector of shape (3,).

    Returns:
        transformed_points (np.ndarray): 3D points in camera coordiante system of shape (N, 2).
    """
    N = points.shape[0]

    # compute world to camera transformation
    T_world2cam = np.eye(4)
    T_world2cam[:3, :3] = R
    T_world2cam[:3, 3] = T

    # convert 3D points to homogeneous coordinates
    points_3D = points.T  # (3, N)
    points_homo = np.vstack([points_3D, np.ones((1, N))])  # (4, N)

    # transform points to the camera frame
    transformed_points = T_world2cam @ points_homo  # (4, N)
    transformed_points = transformed_points[:3, :]  # (3, N)
    transformed_points = transformed_points.T  # (N, 3)

    return transformed_points


def perspective_projection(points, K):
    """ Project 3D points in camera coordinate system onto the camera plane
    Args:
        points (np.ndarray): 3D points in camera coordinate system of shape (N, 3).
        K (np.ndarray): camera intrinsic matrix of shape (3, 3).

    Returns:
        proj_points (np.ndarray): 2D points of shape (N, 2).
    """
    points = points.T  # (3, N)

    # project to image plane
    points_2D = K @ points  # (3, N)
    points_2D = points_2D[:2, :] / points_2D[2, :]  # (2, N)
    proj_points = points_2D.T  # (N, 2)

    return proj_points

def visualize(root_dir, outputpath=None, view_id=0, 
              visualize_smpl=False, smpl_model_path=None,shape_path=None):
    # load color image
    png_files = glob.glob(os.path.join(root_dir, 'images','*.png'))
    # skip=2
    # for i in range(0,646,skip):
    mesh_infos = {}
    cameras_pkl = {}
    smpl_model = SMPL(sex='neutral', model_dir="/root/workspace-NerfHuman/CVPRversionAnimateHuman/third_parties/smpl/models")
    body_SMPL = body_model.SMPLlayer("/root/workspace-NerfHuman/CVPRversionAnimateHuman/third_parties/smpl/models",gender='neutral')
    try:
        smpl_params_path = osp.join(root_dir,  f'poses_optimized.npz')
        smpl_params = np.load(smpl_params_path)
    except:
        print("no poses_optimized.npz found, use poses.npz!")
        smpl_params_path = osp.join(root_dir,  f'poses.npz')
        smpl_params = np.load(smpl_params_path)

    shapes = smpl_params['betas']
    p=np.concatenate([smpl_params['global_orient'],smpl_params['body_pose']],axis=1)
    poses = p
    Th = smpl_params['transl']
    Rh = poses[...,:3].copy()
    # print(Rh.shape)
    zjumocapsmpl = body_SMPL.convert_from_standard_smpl(poses, shapes, Rh=Rh, Th=Th, expression=None)
    poses = zjumocapsmpl['poses']
    shapes = zjumocapsmpl['shapes']
    Rh = zjumocapsmpl['Rh']
    Th = zjumocapsmpl['Th']

    with open(shape_path,'rb') as f:
        tmp = pickle.load(f)
        k0 = list(tmp.keys())[0]
        shapes = tmp[k0]['betas']
    
    png_files.sort(key=lambda x:int(x[-8:-4]))
    for i,filename in tqdm(enumerate(png_files)):
        frame_id = i
        # color_path = osp.join(root_dir,"images", f"frame_{frame_id:06d}.png")
        # print("imagepath:",color_path)
        # color = cv2.imread(color_path)
        
        # if visualize_smpl:

        

        # initialize SMPL body model
        # smpl = smplx.create(
        #     model_path=smpl_model_path,
        #     model_type='smpl',
        #     gender='neutral')
        # idx = int(frame_id)
        # idx = int(frame_id/skip)
        # print(idx)
        # load SMPL parameters in the world coordinate system
        # smpl_params_path = osp.join(root_dir, seq_name,  f'poses.npz')
       
        # # print(list(smpl_params.keys()))
        # # global_orient = smpl_params["thetas"][..., :3]
        # global_orient = smpl_params["global_orient"]
        # global_orient = global_orient[idx]
        # # body_pose = smpl_params["thetas"][..., 3:]
        # body_pose = smpl_params["body_pose"]
        # body_pose = body_pose[idx]
        # betas = smpl_params['betas']
        # transl = smpl_params['transl'][idx]

        # 加一部分把基础smpl变到zjumocap的格式的代码
        
        

        # compute SMPL vertices in the world coordinate system
        # output = smpl(
        #     betas=torch.Tensor(betas).view(1, 10),
        #     body_pose=torch.Tensor(body_pose).view(1, 23, 3),
        #     global_orient=torch.Tensor(global_orient).view(1, 1, 3),
        #     transl=torch.Tensor(transl).view(1, 3),
        #     return_verts=True
        # )
        # vertices = output.vertices.detach().numpy().squeeze()
        # print(output)
        p = poses[i]
        # s = shapes[i]
        s = shapes
        # print(poses.shape,p.shape,shapes,shapes.shape)
        _, tpose_joints = smpl_model(np.zeros_like(p), s)
        vertices, joints = smpl_model(p, s)

        fn = filename.split("/")[-1]
        fn = fn.split(".")[0]
        mesh_infos[fn] = {
                'Rh': Rh[i],
                'Th': Th[i],
                'poses': p,
                'betas':s,
                'joints': joints,
                'tpose_joints': tpose_joints,
                'vertices': vertices
            }
        

        # load camera parameters
        # camera_path = osp.join(root_dir, seq_name, f'view{view_id:1d}','cameras.npz')
        camera_path = osp.join(root_dir,'cameras.npz')
        camera = np.load(camera_path)

        # K, R, T = camera_params['K'], camera_params['R'], camera_params['T']
        K = camera["intrinsic"]
        R = camera["extrinsic"][:3, :3]
        T  = camera["extrinsic"][:3, 3]

        cameras_pkl[fn] = {
                'intrinsics': camera["intrinsic"],
                'extrinsics': camera["extrinsic"],
                'distortions': np.zeros(5)
            }
        # print(camera["intrinsic"],camera["extrinsic"],np.zeros(5))
        
        # # transform the vertices to the camera coordinate system
        # vertices_cam = transform_points(vertices, R, T)

        # # project the vertices
        # proj_vertices = perspective_projection(vertices_cam, K)

        # # draw on the color image
        # proj_vertices = np.round(proj_vertices).astype(int)
        # for point in proj_vertices:
        #     color = cv2.circle(color, point, 1, (255, 255, 255), thickness=-1)
        # # output_image_path = f"./data/zju_mocap/view{view_id:3d}/output-origin/output_{frame_id:06d}.png"
        # output_image_path = osp.join(root_dir,f"output-origin/output_{frame_id:06d}.png")
        # # Create the output directory if it doesn't exist
        # output_dir = os.path.dirname(output_image_path)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # cv2.imwrite(output_image_path, color)
    avg_betas = shapes
    # avg_betas = np.mean(shapes, axis=0)
    # print("avg_betas:",avg_betas.shape)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    # write camera infos
    with open(os.path.join(outputpath, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras_pkl, f)

    # write mesh infos
    with open(os.path.join(outputpath, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)
    tve, template_joints = smpl_model(np.zeros_like(p), avg_betas)
    with open(os.path.join(outputpath, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'joints': template_joints,
                't-pose': np.zeros_like(p),
                'avg_betas': avg_betas,
                't-v': tve,
            }, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='root directory in which data is stored.')
    parser.add_argument('--outputpath', 
                        default="/root/workspace-NerfHuman/CVPRversionAnimateHuman/dataset/custom/lgp1",
                        help='outputpath' )
    # parser.add_argument('view_id', type=int, default=0,choices=list(range(23)),
    #                     help='Kinect ID. Available range is [0, 22]')
    # parser.add_argument('--visualize_smpl', action='store_true',
    #                     help='whether to overlay SMPL vertices on color image.')
    parser.add_argument('--shapepath',
                        default='/root/workspace-NerfHuman/CVPRversionAnimateHuman/dataset/custom/lgp1"',
                        help="directory in which aim SMPL body models are stored")
    parser.add_argument('--smpl_model_path',
                        default='/root/workspace-NerfHuman/humannerf/third_parties',
                        help="directory in which SMPL body models are stored")
    
    args = parser.parse_args()

    visualize(
        root_dir=args.root_dir,
        outputpath=args.outputpath,
        # view_id=args.view_id,
        # visualize_smpl=args.visualize_smpl,
        smpl_model_path=args.smpl_model_path,
        shape_path = args.shapepath,
    )
