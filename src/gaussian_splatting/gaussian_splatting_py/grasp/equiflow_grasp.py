# This is the script that uses Equivariant Grasp Net to grasp objects
# It could be served as a template for other grasp methods

import glob
import os
import argparse
import random
import yaml
import cv2
import torch
import datetime
import numpy as np
import open3d as o3d

from gaussian_splatting_py.datasets import KinovaDataset
from kinova_control_py.gripper_model import Gripper
from kinova_control_py.grasp.equigrasp_model import get_model

from omegaconf import OmegaConf

class EquiFlowGraspNet:
    """ 
        
    """
    def __init__(self, cfg, args):
        """  

            Args:
                mesh_file: the mesh file of the object (CAD Model)
                recorded_poses: the recorded grasping poses of the object
        """
        # Setup model
        self.model = get_model(cfg.model).to(cfg.device)

    def predict_grasp_pose(self, **obs):
        """ 
            Predict the grasp pose of the object
        
            obs: the observation of the object
                dict: 
                    has keys 'data_path'
        """
        # take the first observation
        dataset = KinovaDataset(obs['data_path'])
        data_pack = dataset[0]

        segmap, rgb, depth, cam_K, pc_full, pc_colors = \
                data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], None, None
        H, W, C = rgb.shape
        depth = depth.reshape(H, W)
        segmap, rgb, depth, cam_K = 1 - segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy()

        # dilate the segmap
        # import pdb; pdb.set_trace()
        # segmap = segmap.astype(np.uint8)
        # segmap = cv2.dilate(segmap, np.ones((10,10), np.uint8), iterations=1)

        # build & crop point cloud
        depth = depth * segmap
        pc_full, color = self.depth2pc(depth, cam_K)
        pc_ds = self.downsample_pc(pc_full, num_points=512)

        # compute scale 
        # scale_transform = self.normalize_pc(pc_ds)
        scale_transform = np.eye(4)
        
        # normalize point cloud
        pc_ds_norm = pc_ds @ scale_transform[:3, :3].T + scale_transform[:3, 3]
        pc_ds_norm_t = torch.from_numpy(pc_ds_norm).cuda().float()
        pc_ds_norm_t = pc_ds_norm_t.unsqueeze(0).permute(0, 2, 1)
        
        # read from cad model
        # import pdb; pdb.set_trace()
        # stl = o3d.io.read_triangle_mesh("/home/user/Documents/ActiveTouch/src/gaussian_splatting/weights/CAD_models/can.stl")
        # pc_ds = stl.sample_points_uniformly(number_of_points=512)
        # pc_ds = np.asarray(pc_ds.points)
        # scale_transform = self.normalize_pc(pc_ds)
        
        # pc_ds_norm = pc_ds @ scale_transform[:3, :3].T + scale_transform[:3, 3]
        # pc_ds_norm_t = torch.from_numpy(pc_ds_norm).cuda().float()
        # pc_ds_norm_t = pc_ds_norm_t.unsqueeze(0).permute(0, 2, 1)
        
        nums_grasps = torch.tensor([10], dtype=torch.int32, device=pc_ds_norm_t.device)
        Ts_grasp_views_pred = self.model.sample(pc_ds_norm_t, nums_grasps)
        
        if True:
            world_pcd = o3d.geometry.PointCloud()
            world_pcd.points = o3d.utility.Vector3dVector(pc_ds)

            poses = Ts_grasp_views_pred[0].detach().cpu().numpy()
            inv_transform = np.linalg.inv(scale_transform)
            poses[:, :3, 3] = poses[:, :3, 3] @ inv_transform[:3, :3].T + inv_transform[:3, 3]

            Gripper.visualize_grasp_poses(poses, [world_pcd])
        
    def step(self, **observation):
        """ Interface for the grasp method """
        pass

    def visualize(self, pc, Ts_grasp_views_pred):
        """ Interface for the grasp method """
        pass

    def normalize_pc(self, pc):
        """
        Find the normalize transform  to [-1, 1]
        :param pc: Nx3 point cloud np.array
        :returns: 4x4 normalized transform
        """
        pc_min = np.min(pc, axis=0)
        pc_max = np.max(pc, axis=0)
        pc_center = (pc_min + pc_max) / 2
        pc_scale = np.max(pc_max - pc_min)

        
        transform = np.eye(4)
        transform[:3, :3] = np.eye(3) / pc_scale
        transform[:3, 3] = -pc_center / pc_scale
        return transform

    def downsample_pc(self, pc, num_points = 512):
        """
        Downsample point cloud to num_points

        :param pc: Nx3 point cloud np.array
        :param num_points: number of points to downsample to
 
        :returns: Nx3 downsampled point cloud np.array
        """
        voxel_size = 0.5
        voxel_size_min = 0
        voxel_size_max = 1

        partial_pcd = o3d.geometry.PointCloud()
        partial_pcd.points = o3d.utility.Vector3dVector(pc)

        while True:
            partial_pcd_tmp = partial_pcd.voxel_down_sample(voxel_size)
            num_points_tmp = len(np.asarray(partial_pcd_tmp.points))

            if num_points_tmp - num_points >= 0 and num_points_tmp - num_points < 100:
                break
            else:
                if num_points_tmp > num_points:
                    voxel_size_min = voxel_size
                elif num_points_tmp < num_points:
                    voxel_size_max = voxel_size

                voxel_size = (voxel_size_min + voxel_size_max) / 2

        partial_pcd = partial_pcd_tmp.select_by_index(np.random.choice(num_points_tmp, num_points, replace=False))
        return np.asarray(partial_pcd.points)

    def depth2pc(self, depth, K, rgb=None):
        """
        Convert depth and intrinsics to point cloud and optionally point cloud color
        :param depth: hxw depth map in m
        :param K: 3x3 Camera Matrix with intrinsics
        :returns: (Nx3 point cloud, point cloud color)
        """

        mask = np.where(depth > 0)
        x,y = mask[1], mask[0]
        
        normalized_x = (x.astype(np.float32) - K[0,2])
        normalized_y = (y.astype(np.float32) - K[1,2])

        world_x = normalized_x * depth[y, x] / K[0,0]
        world_y = normalized_y * depth[y, x] / K[1,1]
        world_z = depth[y, x]

        if rgb is not None:
            rgb = rgb[y,x,:]
            
        pc = np.vstack((world_x, world_y, world_z)).T
        return (pc, rgb)

def main(args, cfg):
    seed = cfg.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True

    model = EquiFlowGraspNet(cfg, args)
    pred = model.predict_grasp_pose(data_path=args.datadir)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_result_path', type=str, default="/home/user/Documents/EquiGraspFlow/checkppoints/equigraspflow_partial")
    parser.add_argument('--checkpoint', type=str, default="model_best_val_loss.pkl")
    parser.add_argument('--device', default=0)
    parser.add_argument('--logdir', default='test_results')
    parser.add_argument('--datadir', type=str, default='/home/user/Documents/data/red_cup')

    args = parser.parse_args()

    # Load config
    config_filename = [file for file in os.listdir(args.train_result_path) if file.endswith('.yml')][0]
    cfg = OmegaConf.load(os.path.join(args.train_result_path, config_filename))

    # Setup checkpoint
    cfg.model.checkpoint = os.path.join(args.train_result_path, args.checkpoint)

    # Setup device
    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    main(args, cfg)

    