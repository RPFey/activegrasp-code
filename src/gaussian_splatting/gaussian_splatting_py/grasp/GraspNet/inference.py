import torch
import os
import random
import sys
import numpy as np
import argparse
import time
import open3d as o3d
import MinkowskiEngine as ME
import cv2

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
from gaussian_splatting_py.grasp.GraspNet.model import GraspNet_MSCQ, GraspNet1
from gaussian_splatting_py.grasp.GraspNet.utils import CameraInfo, create_point_cloud_from_depth_image
from gaussian_splatting_py.grasp.base import GraspEstimator, extract_point_clouds, extract_object_pc

import gaussian_splatting_py
PACKAGE_ROOT = os.path.join(gaussian_splatting_py.__path__[0], "../")
WEIGHTS_ROOT = os.path.join(PACKAGE_ROOT, "weights")
CONFIG_ROOT = os.path.join(PACKAGE_ROOT, "config")

class GraspnessPredictor:
    def __init__(self, cfg, eslam):
        self.net = GraspNet1(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                    cylinder_radius=0.08, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        checkpoint_path = os.path.join(WEIGHTS_ROOT, cfg['grasp_checkpoint'])
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        self.num_points = 50000
        self.voxel_size = 0.005
        self.cfg = cfg


    def process_input(self, depth):
        mask = (depth > 0)
        camera = CameraInfo(self.cfg['cam']['W'], self.cfg['cam']['H'], self.cfg['cam']['fx'], self.cfg['cam']['fy'], self.cfg['cam']['cx'], self.cfg['cam']['cy'],1)

        #mask2 = np.zeros_like(depth)
        point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        cloud_masked = point_cloud[mask]
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked#[idxs]
        ret_dict = {}
        ret_dict['point_clouds'] = torch.from_numpy(cloud_sampled.astype(np.float32)).unsqueeze(0).to(self.device)
        coors = cloud_sampled.astype(np.float32) / self.voxel_size
        feats = np.ones_like(cloud_sampled).astype(np.float32)
        coordinates_batch, features_batch = ME.utils.sparse_collate([coors],[feats])
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
        ret_dict['coors'] = coordinates_batch.to(self.device)
        ret_dict['feats'] = features_batch.to(self.device)
        ret_dict['quantize2original'] = quantize2original.to(self.device)
        ret_dict['mask'] = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return ret_dict

    def process_pointcloud(self, point_cloud):

        cloud_masked = point_cloud
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked#[idxs]
        ret_dict = {}
        ret_dict['point_clouds'] = torch.from_numpy(cloud_sampled.astype(np.float32)).unsqueeze(0).to(self.device)
        coors = cloud_sampled.astype(np.float32) / self.voxel_size
        feats = np.ones_like(cloud_sampled).astype(np.float32)
        coordinates_batch, features_batch = ME.utils.sparse_collate([coors],[feats])
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
        ret_dict['coors'] = coordinates_batch.to(self.device)
        ret_dict['feats'] = features_batch.to(self.device)
        ret_dict['quantize2original'] = quantize2original.to(self.device)
        return ret_dict

    def inference(self, depth):
        #todo: make it differentiable
        H, W = depth.shape
        end_points = self.process_input(depth.squeeze(0).detach().cpu().numpy())
        with torch.no_grad():
            end_points = self.net(end_points)
        graspness_score = end_points['graspness_score'].squeeze()
        objectness_mask = end_points['objectness_mask'].squeeze()
        graspness_score[~objectness_mask] = 0
        mask = end_points['mask'].squeeze()
        gt_grasness = torch.zeros((H, W)).float().to(self.device, non_blocking=True)
        gt_grasness[mask] = graspness_score
        gt_objectness = torch.zeros((H, W)).bool().to(self.device, non_blocking=True)
        gt_objectness[mask] = 1 # objectness_mask
        gt_grasness = gt_grasness.unsqueeze(0)
        gt_objectness = gt_objectness.unsqueeze(0)
        return gt_grasness, gt_objectness

    def view_affordance(self, point_cloud, c2w):
        end_points = self.process_pointcloud(point_cloud)
        with torch.no_grad():
            end_points = self.net(end_points)
        view_score = end_points["view_score"]
        B, num_seed, _ = view_score.shape # (B, num_seed, num_view)
        transform_x = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float().to(view_score.device)
        trans = torch.matmul(c2w, transform_x)
        world_approach_xyz = torch.tensor([0, 0, 1]).float().to(view_score.device)
        template_views = generate_grasp_views(300).to(view_score.device)  # (num_view, 3)

        view_sim = torch.cosine_similarity(trans[2, :3], template_views, dim=-1)  # (B,300)
        view_idx = torch.argmax(view_sim,dim=-1)
        view_idx = view_idx.view(1, 1, 1).expand(-1, num_seed, -1).contiguous()
        candidate_view_score = torch.gather(view_score, 2, view_idx).squeeze()  # (B, num_seed)
        return torch.mean(candidate_view_score)


def generate_grasp_views(N=300, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return torch.from_numpy(views.astype(np.float32))

if __name__ == "__main__":
    from gaussian_splatting_py.grasp.GraspNet.config import load_config
    cfg = load_config(os.path.join(CONFIG_ROOT, 'GraspNet', 'scene_0100.yaml'), 
                      os.path.join(CONFIG_ROOT, 'ESLAM.yaml'))
    grasp_predictor = GraspnessPredictor(cfg, None)
    depth_path = '/root/ActiveGrasp/tmp/2025-04-19-19-40-47-s12-ep2/depths/0005.png'
    depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) 
    depth_data = cv2.resize(depth_data, (cfg['cam']['W'], cfg['cam']['H']))
    depth_data = depth_data.astype(np.float32) / 1000.
    depth_data = torch.from_numpy(depth_data).to(grasp_predictor.device)
    res = grasp_predictor.inference(depth_data)

    import pdb; pdb.set_trace()
    pass