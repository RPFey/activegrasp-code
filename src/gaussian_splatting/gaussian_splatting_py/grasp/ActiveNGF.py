import argparse
import os
import random
import sys
import time

import cv2
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from graspnetAPI.graspnet_eval import GraspGroup
from sksparse.cholmod import Factor

from contact_graspnet_pytorch.visualization_utils_o3d import show_image, visualize_grasps
from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.base import (
    GraspEstimator,
    extract_object_pc,
    extract_pcd_batch,
    extract_point_clouds,
)
from gaussian_splatting_py.grasp.GraspNet.config import load_config
from gaussian_splatting_py.grasp.GraspNet.ESLAM import ESLAM
from gaussian_splatting_py.grasp.GraspNet.graspnet import (
    GraspNet,
    minkowski_collate_fn,
    pred_decode,
)
from gaussian_splatting_py.grasp.GraspNet.model import GraspNet_MSCQ, pred_decode_reg

import gaussian_splatting_py
PACKAGE_ROOT = os.path.join(gaussian_splatting_py.__path__[0], "../")
WEIGHTS_ROOT = os.path.join(PACKAGE_ROOT, "weights")
CONFIG_ROOT = os.path.join(PACKAGE_ROOT, "config")

class ActiveNGFEstimator(GraspEstimator):
    def __init__(self, exp_name, num_grasps=256):
        cfg = load_config(os.path.join(CONFIG_ROOT, 'GraspNet', 'scene_0100.yaml'), 
                      os.path.join(CONFIG_ROOT, 'ESLAM.yaml'))
        
        self.cfg = cfg
        self.system = ESLAM(cfg, exp_name)
        self.frame_nums = 0
        self.object_center = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Generalized 6D Grasp Predictor, 
        # In the paper, they say they use the grasp prediction from Generalized 6D Grasp Predictor
        # self.grasp_predictor = GraspNet_MSCQ(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        #             cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        # self.grasp_predictor.to(self.device)
        # self.grasp_predictor.eval()
        # model_weights = torch.load(os.path.join(WEIGHTS_ROOT, 'log_phy/checkpoint.tar'), map_location=self.device)
        # self.grasp_predictor.load_state_dict(model_weights['model_state_dict'], strict=True)
        
        # graspnet predictor
        self.grasp_predictor = GraspNet(seed_feat_dim=512, is_training=False)
        self.grasp_predictor.to(self.device)
        checkpoint = torch.load(
            os.path.join(WEIGHTS_ROOT, 'minkuresunet_realsense.tar'), map_location=self.device)
        self.grasp_predictor.load_state_dict(checkpoint['model_state_dict'])
        
        self.num_grasps = num_grasps
        
    def resample_images(self, rgb, depth, segmap, cam_K):
        """
            Resample the images to the desired intrinsic.
            Args:
                rgb: [numpy.ndarray, (H,W,3), numpy.uint8]
                    RGB image
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    Depth image
                segmap: [numpy.ndarray, (H,W), numpy.uint8]
                    Segmentation map
                cam_K: [numpy.ndarray, (3,3), numpy.float32]
                    Camera intrinsic matrix
            Returns:
                rgb: [numpy.ndarray, (H,W,3), numpy.uint8]
                    Resampled RGB image
                depth: [numpy.ndarray, (H,W), numpy.float32]
                    Resampled Depth image
                segmap: [numpy.ndarray, (H,W), numpy.uint8]
                    Resampled Segmentation map
                cam_K: [numpy.ndarray, (3,3), numpy.float32]
        """
        desired_K = torch.eye(3)
        desired_K[0, 0] = self.cfg['cam']['fx']
        desired_K[1, 1] = self.cfg['cam']['fy']
        desired_K[0, 2] = self.cfg['cam']['cx']
        desired_K[1, 2] = self.cfg['cam']['cy']
        desired_W, desired_H = self.cfg['cam']['W'], self.cfg['cam']['H']
        grid_points = torch.meshgrid(torch.arange(desired_W), torch.arange(desired_H))
        grid_points = torch.stack(grid_points, axis=-1).reshape(-1, 2)
        grid_points = torch.cat([grid_points, torch.ones((grid_points.shape[0], 1))], axis=-1)
        grid_points = torch.matmul(grid_points, torch.linalg.inv(desired_K).T)
        grid_points = grid_points.reshape(desired_H, desired_W, 3)
        original_W, original_H = rgb.shape[1], rgb.shape[0]
        
        # project the points to the original image
        original_points = torch.matmul(grid_points, cam_K.T)
        original_points = original_points.reshape(-1, 3)
        original_points = original_points / original_points[:, 2:]
        original_points = original_points[:, :2]
        
        # sample the points on the original image
        original_points = original_points.reshape(desired_H, desired_W, 2)
        # original_points = np.clip(original_points, 0, np.array([rgb.shape[1]-1, rgb.shape[0]-1]))
        original_points[..., 0] = 2 * original_points[..., 0] / original_W - 1
        original_points[..., 1] = 2 * original_points[..., 1] / original_H - 1
        
        # use pytorch grid sample
        original_points = original_points.float()
        original_points.clamp_(min=-1, max=1)
        # original_points = original_points.permute(2, 0, 1).unsqueeze(0)
        original_points = original_points.to(rgb.device).unsqueeze(0)
        
        # concatenate together, then grid sample
        features = torch.cat([rgb.permute(2, 0, 1), depth.unsqueeze(0), segmap.unsqueeze(0)], dim=0)
        features = features.unsqueeze(0)
        features = features.to(rgb.device)
        
        features = torch.nn.functional.grid_sample(features, original_points, mode='bilinear', align_corners=True)
        features = features.squeeze(0).permute(1, 2, 0)
        
        # split to rgb, depth, segmap
        rgb = features[:, :, :3]
        depth = features[:, :, 3]
        segmap = features[:, :, 4]
        
        return rgb, depth, segmap, desired_K
        
    def __mapping(self, **obs):
        # take the first observation
        if 'data_path' in obs:
            dataset = KinovaDataset(obs['data_path'])
        else:
            dataset = obs['dataset']
            
        # compute the first frame
        if self.object_center is None:
            object_pcs = extract_object_pc(dataset)[1]
            min_coord, max_coord = np.min(object_pcs, axis=0), np.max(object_pcs, axis=0)
            self.object_center = (min_coord + max_coord) / 2
            self.object_center = torch.from_numpy(self.object_center).float()
        
        total_frames = len(dataset)
        for idx in range(self.frame_nums, total_frames):
            
            data_pack = dataset[idx]
            segmap, rgb, depth, cam_K, c2w = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], data_pack['camtoworld']
                    
            H, W = rgb.shape[:2]
            depth = depth.reshape(H, W)
            
            _rgb, _depth, _segmap, _ = self.resample_images(rgb, depth, segmap, cam_K)
            _rgb = _rgb / 255.
            c2w_center = c2w.clone()
            c2w_center[:3, 3] -= self.object_center.to(c2w.device)
            
            # change to x - right; y - up, z - back
            transform_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float().to(c2w.device)
            c2w_center = torch.matmul(c2w_center, transform_x)
            
            # update maps
            self.system.mapper.mapping_step(idx, _rgb, _depth, c2w_center)
            self.frame_nums += 1
            
    def process_input(self, pc_full, fuse_pc_segments, outlier=0.02, num_points=15000, voxel_size=0.005):
        """
            Dataset Process for GraspNet input
        
            I follow the dataset process for GraspNet in the paper.
                the foreground points are points on the target object.
        
        Args:
            pc_full, fuse_pc_segments: Output from extract pcds from dataset.
            
        Return:
            ret_dict: the dictionary of the point cloud and the features
                point_clouds: the point cloud (N, 3)
                coors: the coordinates of the point cloud in voxels (N, 3)
                feats: the features of the point cloud (N, 1)
        """
    
        # cropping the points to the workspace (target object)
        foreground = fuse_pc_segments[1] # only the object
        xmin, ymin, zmin = foreground.min(axis=0)
        xmax, ymax, zmax = foreground.max(axis=0)
        mask_x = ((pc_full[:, 0] > xmin - outlier) & (pc_full[:, 0] < xmax + outlier))
        mask_y = ((pc_full[:, 1] > ymin - outlier) & (pc_full[:, 1] < ymax + outlier))
        mask_z = ((pc_full[:, 2] > zmin - outlier) & (pc_full[:, 2] < zmax + outlier))
        workspace_mask = (mask_x & mask_y & mask_z)
        workspace_pc = pc_full[workspace_mask]
        
        # Use only the foreground
        workspace_pc = foreground
        
        # Sample the points to num_points
        if len(workspace_pc) >= num_points:
            idxs = np.random.choice(len(workspace_pc), num_points, replace=False)
        else:
            idxs1 = np.arange(len(workspace_pc))
            idxs2 = np.random.choice(len(workspace_pc), num_points - len(workspace_pc), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = workspace_pc[idxs]
                            
        ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32),
        }
        
        # pre-process the point cloud for Minkowski Engine
        batch_data = minkowski_collate_fn([ret_dict])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        
        return batch_data
        
    def predict_grasp_pose(self, **obs):
        # get the dataset
        if 'data_path' in obs:
            dataset = KinovaDataset(obs['data_path'])
        else:
            dataset = obs['dataset']
            
        # Include the scene information
        pc_full, pc_colors, fuse_pc_segments, base_transform = \
                            extract_pcd_batch(dataset, use_prot_comp=False, include_scene_info=True)
        
        # Run GraspNet Inference
        batch_data = self.process_input(pc_full, fuse_pc_segments)
        with torch.no_grad():
            end_points = self.grasp_predictor(batch_data)
            grasp_preds = pred_decode(end_points)
            preds = grasp_preds[0].detach().cpu().numpy()
            
            # convert to GraspGroup and do nms
            gg = GraspGroup(preds)
            gg = gg.nms()
            gg = gg.sort_by_score()
            if len(gg) > self.num_grasps:
                gg = gg[:self.num_grasps]
        
        offset_x = 0.0584 + 0.02 + 0.004 # tail length + depth base + finger width.
        offset = np.array([
            [0, 0, 1, -1 * offset_x],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        
        scores, poses = [], []
        for g in gg:
            scores.append(g.score)
            T = np.eye(4)
            T[:3, :3] = g.rotation_matrix
            T[:3, 3] = g.translation
            poses.append( T @ offset )
        poses = np.stack(poses, axis=0)
        scores = np.stack(scores)
        
        if obs.get('viz', False):
            
            # visualize the grasps using GraspGroup
            # pcd = o3d.geometry.PointCloud()
            # pcd_points = batch_data['point_clouds'].cpu().numpy().reshape(-1, 3)
            # pcd.points = o3d.utility.Vector3dVector(pcd_points)
            # grippers = gg.to_open3d_geometry_list()
            # o3d.visualization.draw_geometries([pcd] + grippers)
            
            pc_full, pc_colors = pc_full[::8], pc_colors[::8]
            visualize_grasps(pc_full, {1: poses}, {1: scores}, pc_colors=pc_colors)
            
            
        top_k_poses = np.stack([base_transform @ g for g in poses], axis=0)
        return top_k_poses, {"scores": scores}
            
    def compute_IG(self, c2ws, dataset):
        self.__mapping(dataset=dataset)
        
        # compute the uncertainty scores.
        scores = []
        for i, pose in enumerate(c2ws):
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float()
            
            # preprocess the camera view poses
            c2w_center = pose.clone()
            c2w_center[:3, 3] -= self.object_center.to(c2w_center.device)
            transform_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float().to(c2w_center.device)
            c2w_center = torch.matmul(c2w_center, transform_x)
            
            try:
                uncertainty, graspness, net_graspness, depth, \
                        objectness_mask = self.system.mapper.uncertainty_estimation(c2w_center.to(self.device))
                scores.append(uncertainty.item())
            except RuntimeError as e:
                print(f"Error in uncertainty estimation: {e}")
                scores.append(0)
        
        return np.array(scores)
                     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--data_path', type=str, default='/root/ActiveGrasp/tmp/2025-04-19-19-40-47-s12-ep2')
    parser.add_argument('--viz', action='store_true', default=False, help='visualize the grasps')
    args = parser.parse_args()
    
    estimator = ActiveNGFEstimator(args.exp_name, 16)
    # Add code to use the estimator for grasp prediction
    estimator.predict_grasp_pose(data_path=args.data_path, viz=args.viz)


