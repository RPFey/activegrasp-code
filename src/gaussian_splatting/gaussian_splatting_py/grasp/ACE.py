import os
import json
import shutil
from typing import Any
import torch
import hydra
import argparse
import itertools
import cv2
import numpy as np
from tqdm import tqdm
from hydra import compose, initialize
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, List, Dict, Set, Tuple, Union
from datetime import datetime
import logging

# from vgn.io import my_IO
# from vgn.perception import *
# from vgn.simulation import ClutterRemovalSim
# from vgn.utils.misc import apply_noise

from gaussian_splatting_py.grasp.base import GraspEstimator, extract_point_clouds, extract_object_pc
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image

# imports from VGN Part
from gaussian_splatting_py.grasp.vgn_src.grasp import Grasp, Label
from gaussian_splatting_py.grasp.vgn_src.transform import Rotation, Transform
from gaussian_splatting_py.grasp.vgn_src.perception import TSDFVolume, CameraIntrinsic

from gaussian_splatting_py.grasp.ace_src.networks import Runner
from gaussian_splatting_py.grasp.ace_src.cam_transform import pixel2world, world2pixel
from gaussian_splatting_py.grasp.ace_src.ACE_architecture import EasyDict

# from utils.misc import EasyDict, set_random_seed, blockPrint, enablePrint
# from utils.visualization import *
# from utils.transform import *

def _get_random_angles(size=0.3) -> Tuple[float, float, float]:
    '''
        Generate random angles for grasp pose
    '''
    r = np.random.uniform(1.6, 2.4) * size
    theta = np.random.uniform(np.pi / 4.0, np.pi / 2.0-0.2)  # test
    # theta = np.random.uniform(0.0, np.pi / 3.0)  # default pi/4
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    return r, theta, phi

def _extrinsic_to_angles(transform: Transform, size=0.3) -> Tuple[float, float, float]:
    '''extrinsic matrix to sphereical coordinate
    '''
    origin = Transform(Rotation.identity(), np.r_[size / 2, size / 2, 0.0])
    m = (transform * origin).inverse().as_matrix()

    eye = m[:3, 3]
    forward = m[:3, 2]
    right = m[:3, 0]
    up = -m[:3, 1]

    radius = np.linalg.norm(eye)
    theta = np.arccos(eye[2] / radius)
    phi = np.arctan2(eye[1], eye[0])

    if phi < 0.0:
        phi += 2.0 * np.pi

    return radius, theta, phi

import torch

def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions to rotation matrices.
    Quaternion format: [x, y, z, w]

    Args:
        quat (torch.Tensor): Tensor of shape (N, 4)

    Returns:
        torch.Tensor: Rotation matrices of shape (N, 3, 3)
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Normalize the quaternion to ensure it represents a valid rotation
    norm = torch.sqrt(x*x + y*y + z*z + w*w + 1e-8)
    x = x / norm
    y = y / norm
    z = z / norm
    w = w / norm

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rot = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw),
        2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)
    ], dim=1).reshape(-1, 3, 3)

    return rot


def get_ckpts(experiment_dir, only_epoch_end=False):
    '''获取某个实验的所有 ckpt list
    only_epoch_end: 只用每个 epoch end 保存的 ckpt
    '''
    ckpts_path = os.path.join(experiment_dir, 'ckpts')
    ckpts = [f for f in os.listdir(ckpts_path)
             if not only_epoch_end or f.endswith('end.pt')]
    # ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(ckpts_path, f)))
    ckpts.sort()
    return [os.path.join(ckpts_path, f) for f in ckpts]

def load_hydra_cfg(experiment_dir):
    '''load configs from experiment log, and override ckpt_name, device, etc.
    '''
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    dirname = os.path.dirname(__file__)
    config_path = os.path.join(dirname, "../../weights/", experiment_dir, "configs")
    rel_config_path = os.path.relpath(config_path, dirname)
    logging.debug(f"rel_config_path: {rel_config_path}")
    initialize(version_base=None,
               config_path=rel_config_path,
               job_name="test_app")
    # initialize(version_base=None,
    #            config_path=os.path.join(dirname, "../../weights/", experiment_dir, "configs"),
    #            job_name="test_app")
    ckpt_name = get_ckpts(os.path.join(dirname, "../../weights/", experiment_dir))[-1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cfg = compose(config_name="config", overrides=[f"load_path='{ckpt_name}'",
                                                   f"device={device}"])
    return cfg

class AABBox:
    def __init__(self, bbox_min, bbox_max):
        self.min = np.asarray(bbox_min)
        self.max = np.asarray(bbox_max)
        self.center = 0.5 * (self.min + self.max)
        self.size = self.max - self.min

    @property
    def corners(self):
        return list(itertools.product(*np.vstack((self.min, self.max)).T))

    def is_inside(self, p):
        return np.all(p > self.min) and np.all(p < self.max)

class GraspPlanner:
    '''
    Grasp Planner, including network interface, data preparation, and post-processing
    '''
    def __init__(self, cfg):
        self.net = Runner(cfg)

    def __call__(self, tsdf, pos_list, extrinsic_list, intrinsic_list, size) -> EasyDict:
        """ Call the grasp planner 
        
        Args:
            tsdf: TSDF Volume in shape (1, resolution**3), np.ndarray
            pos_list: List[array], 3D Candidate Position
            extrinsic_list: array, camera extrinsic
            intrinsic_list: array, camera intrinsic
        """
        result = EasyDict()
        
        if len(pos_list) == 0:
            result.grasp_label = torch.tensor([0])
            return result
        
        batch_size = 1000  # 一次预测的点数量 为了保显存
        is_first_batch = True
        
        for i in tqdm(range(0, len(pos_list), batch_size), disable=True):  # 分批处理
            batch = self._prepare_batch(tsdf,
                                        pos_list[i:i + batch_size],  # 索引会自动处理最后一批不足 batch_size 的情况
                                        extrinsic_list,
                                        intrinsic_list,
                                        size)
            prediction_batch = self.net.predict_grasp(batch)
            prediction_batch = self._clean_prediction(prediction_batch, size)
            if is_first_batch:
                result = prediction_batch
                is_first_batch = False
            else:
                result.append(prediction_batch)  # feature from EasyDict
        return result

    def _prepare_batch(self, tsdf, pos_list, extrinsics, intrinsics, size) -> EasyDict:
        '''将数据转换为网络输入格式, 空间点 normalize
        tsdf: [1, resolution**3]
        pos_list: [n, 1, 3]
        extrinsics: [1, 7]
        intrinsics: [1, 6]
        '''
        # 数据都转成 tensor
        tsdf = tsdf if isinstance(tsdf, torch.Tensor) else torch.tensor(tsdf)
        pos_list = pos_list if isinstance(pos_list, torch.Tensor) else torch.tensor(pos_list)
        if len(pos_list.shape) == 2:
            pos_list = pos_list.unsqueeze(1)
        extrinsics = extrinsics if isinstance(extrinsics, torch.Tensor) else torch.tensor(extrinsics)
        intrinsics = intrinsics if isinstance(intrinsics, torch.Tensor) else torch.tensor(intrinsics)

        batch_size = len(pos_list)
        assert batch_size != 0
        # tsdf = tsdf.repeat(batch_size, 1, 1, 1)  # 网络自动处理 只需要 encode 一次
        extrinsics = extrinsics[None, None, :].repeat(batch_size, 1, 1)
        intrinsics = intrinsics[None, :].repeat(batch_size, 1)

        # normalize
        pos_list = pos_list / size - 0.5  # [0.0, 0.3] normalize to [-0.5, 0.5]

        batch = EasyDict({'tsdf': tsdf, 'point_grasp': pos_list,
                          'camera_extrinsic': extrinsics,
                          'camera_intrinsic': intrinsics})
        return batch

    def _clean_prediction(self, prediction: EasyDict, size) -> EasyDict:
        '''恢复到世界坐标尺寸
        '''
        # prediction.
        # pose.translation = (pose.translation + 0.5) * size
        prediction.grasp_width *= size
        return prediction
    
class ACEGraspEstimator(GraspEstimator):
    def __init__(self, exp_name, num_grasps = 256, greedy=False):
        cfg = load_hydra_cfg(exp_name)
        self.grasp_planner = GraspPlanner(cfg)
        self.greedy = greedy
        self.num_grasps = num_grasps
        
        self.tsdf_size = 6 * 0.05
        self.resolution = 40
        
    def predict_grasp_pose(self, **obs):
        # take the first observation
        if 'data_path' in obs:
            dataset = KinovaDataset(obs['data_path'])
        else:
            dataset = obs['dataset']
        
        # compute the first frame
        object_pcs = extract_object_pc(dataset)[1]
        min_coord, max_coord = np.min(object_pcs, axis=0), np.max(object_pcs, axis=0)
        # set the ground to -0.04 to include the table
        min_coord[2] = -0.04
        # bound = np.max(max_coord - min_coord)
        
        tsdf_volume = TSDFVolume(self.tsdf_size, self.resolution, min_coord)
        num_views = len(dataset)
        candidate_grasps = []
        candidate_scores = []
        for i in range(num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K, c2w = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], data_pack['camtoworld']
                    
            if isinstance(segmap, torch.Tensor):
                segmap, rgb, depth, cam_K, c2w = \
                    segmap.cpu().numpy(), rgb.cpu().numpy(), depth.cpu().numpy(), \
                            cam_K.cpu().numpy(), c2w.cpu().numpy()
            
            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            
            camera_intrinsic = CameraIntrinsic(W, H, cam_K[0, 0], cam_K[1, 1], 
                                                    cam_K[0, 2], cam_K[1, 2])
            
            w2c = np.linalg.inv(c2w)
            tsdf_volume.integrate(depth, camera_intrinsic, w2c)
            tsdf_grid = tsdf_volume.get_grid()
            assert tsdf_grid.max() > 0, "TSDF grid is empty, please check the input data"

            pos_list, pixel_list = self.get_target_pos(
                segmap, depth, w2c, cam_K, sample_points_per_view=200)
            
            if len(pos_list) == 0:
                continue
            
            pos_list_centered = np.stack(pos_list, axis=0) - min_coord
            c2w_centered = c2w.copy()
            c2w_centered[:3, 3] -= min_coord
            w2c_centered = np.linalg.inv(c2w_centered)
            extrinsic_list = Transform.from_matrix(w2c_centered).to_list()
            
            
            intrinsic_list = [cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2], W, H]
            prediction = self.grasp_planner(tsdf_grid, pos_list_centered,
                                            extrinsic_list,
                                            intrinsic_list,
                                            self.tsdf_size)
            
            scores = prediction.grasp_label.cpu().numpy() # .cpu().numpy()
            rot = quaternion_to_matrix(prediction.grasp_rotation).cpu().numpy()
            grasp_poses = np.tile(np.eye(4, 4), (len(rot), 1, 1))
            grasp_poses[:, :3, :3] = rot
            grasp_poses[:, :3, 3] = pos_list
            
            candidate_grasps.append(grasp_poses)
            candidate_scores.append(scores)
            
            if self.greedy and scores.max() > 0.92:
                print(f"Found a grasp with score {scores.max()}")
                break
            
        if len(candidate_grasps) == 0:
            print("No grasps found")
            return [], {}
        
        candidate_grasps = np.concatenate(candidate_grasps, axis=0)
        candidate_scores = np.concatenate(candidate_scores, axis=0)
        
        # select the top 512 grasps according to scores
        top_indices = np.argsort(candidate_scores)[-self.num_grasps:]
        candidate_grasps = candidate_grasps[top_indices]
        candidate_scores = candidate_scores[top_indices]
        
        addons = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # 0.05 - Finger Depth; 0.04 CoM offset in URDF
        candidate_grasps[:, :3, 3] -= candidate_grasps[:, :3, 2] * ( 0.05 + 0.04 )
        candidate_grasps = candidate_grasps @ np.linalg.inv(addons)
        
        # visualize grasps
        if obs.get("viz", False):
            data_pack = dataset[0]
            segmap, rgb, depth, cam_K, c2w = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], data_pack['camtoworld']
            
            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                    skip_border_objects=True, 
                                                                    z_range=[0, 2.])   
            pc_full = pc_full @ c2w[:3, :3].T + c2w[:3, 3:4].T             
            visualize_grasps(pc_full, {1: candidate_grasps}, {1: candidate_scores}, pc_colors=pc_colors)
            
        return candidate_grasps, {"scores": candidate_scores}
    
    def compute_IG(self, c2ws, dataset):
        """ 
        Function to compute Information Gain based on Breyer's paper
            https://arxiv.org/pdf/2207.10543
        
        Args:
            c2ws: [np.ndarray] the views of the object (N, 4, 4), c2ws
            dataset: [KinovaDataset] the dataset of the object
            
        Returns:
            scores: [np.ndarray] the scores of the views
        """
        
        # construct TSDF from dataset
        # compute the first frame
        object_pcs = extract_object_pc(dataset)[1]
        min_coord, max_coord = np.min(object_pcs, axis=0), np.max(object_pcs, axis=0)
        # set the ground to -0.04 to include the table
        min_coord[2] = -0.04
        # bound = np.max(max_coord - min_coord)

        tsdf_volume = TSDFVolume(self.tsdf_size, self.resolution, min_coord)
        num_views = len(dataset)
        for i in range(num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K, c2w = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], data_pack['camtoworld']
                    
            if isinstance(segmap, torch.Tensor):
                segmap, rgb, depth, cam_K, c2w = \
                    segmap.cpu().numpy(), rgb.cpu().numpy(), depth.cpu().numpy(), \
                            cam_K.cpu().numpy(), c2w.cpu().numpy()
            
            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            
            camera_intrinsic = CameraIntrinsic(W, H, cam_K[0, 0], cam_K[1, 1], 
                                                    cam_K[0, 2], cam_K[1, 2])
            
            w2c = np.linalg.inv(c2w)
            tsdf_volume.integrate(depth, camera_intrinsic, w2c)
        
        tsdf_grid = tsdf_volume.get_grid()
        
        # compute IG for each view
        scores = []
        cam_K = dataset[0]['K'].cpu().numpy()
        intrinsic_list = [cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2], W, H]
        t_box = np.stack([min_coord, max_coord], axis=0)
        for c2w in tqdm(c2ws, desc="Computing IG"):
            w2c = np.linalg.inv(c2w)
            c2w_centered = c2w.copy()
            c2w_centered[:3, 3] -= min_coord
            w2c_centered = np.linalg.inv(c2w_centered)
            extrinsic_list = Transform.from_matrix(w2c_centered).to_list()
            
            # here we use the last segmeap
            pos_list, pixel_list = self.get_target_pos_from_box(
                t_box, w2c, cam_K, sample_points_per_view=200)
            pos_list_centered = np.stack(pos_list, axis=0) - min_coord
            
            prediction = self.grasp_planner(tsdf_grid, pos_list_centered,
                                            extrinsic_list, intrinsic_list, self.tsdf_size)
            
            max_score = prediction.grasp_label.max().item()
            # select the highest score 
            scores.append(max_score)
        
        return scores
    
    def get_target_pos(self, 
                       target_mask, 
                       depth_img,
                       extrinsic, 
                       intrinsic,
                       sample_points_per_view = 200) -> Tuple[List, List]:
        '''
        Args:
            target_mask: [H, W], 2D mask, 0 - background, 1 - target
            depth_img: [H, W], 2D depth image
            extrinsic: [4, 4], camera extrinsic, w2c
            intrinsic: [3, 3], camera intrinsic
        Return:
            pos_lisy
        '''
        pixel_list = np.argwhere(target_mask)
        # 随机选部分点分 (参数控制)
        if len(pixel_list) > sample_points_per_view:
            pixel_list = pixel_list[np.random.choice(len(pixel_list), sample_points_per_view, replace=False)]

        # pixel -> pos
        pos_list = []
        for y, x in pixel_list:
            z = depth_img[y, x]  # 深度图上的深度
            pos = pixel2world(x, y, z, intrinsic, extrinsic)
            pos_list.append(pos)

        pos_list, pixel_list = self._expand_grasp_depth(
                        pos_list, list(pixel_list), extrinsic=Transform.from_matrix(extrinsic))

        return pos_list, pixel_list
    
    def get_target_pos_from_box(self, t_bbox, 
                                extrinsic, intrinsic, 
                                num_sample_axis=10, sample_points_per_view=200):
        """
        Args:
            box_bounds: [2, 3], 2D box bounds, [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            extrinsic: [4, 4], camera extrinsic, w2c
            intrinsic: [3, 3], camera intrinsic
        Return:
            pos_list: [N, 3], 3D position list
            pixel_list: [N, 2], 2D pixel list
        """
        pos_list = []
        pixel_list = []

        # 给定物体 bbox, 随机选取 bbox 内的点
        x_range = np.linspace(t_bbox[0, 0], t_bbox[1, 0], num_sample_axis)
        y_range = np.linspace(t_bbox[0, 1], t_bbox[1, 1], num_sample_axis)
        z_range = np.linspace(t_bbox[0, 2], t_bbox[1, 2], num_sample_axis)
        for point in itertools.product(x_range, y_range, z_range):
            pos_list.append(point)
            y, x = world2pixel(*point, intrinsic, extrinsic)
            pixel_list.append([x, y])

        # 从 pos 中随机选取 num_pts 个点
        if len(pos_list) > sample_points_per_view:
            sample_index = np.random.choice(len(pos_list), sample_points_per_view, replace=False)
            pos_list = list(np.array(pos_list)[sample_index])
            pixel_list = list(np.array(pixel_list)[sample_index])
        
        return pos_list, pixel_list
        
    def _expand_grasp_depth(self, pos_list: List, pixel_list: List,
                           extrinsic: Transform, finger_depth=0.05, depth_candidates=5) -> Tuple[List, List]:
        '''
            叠加抓取深度
                finger_depth: 默认值在 sim.gripper.finger_depth
        '''
        camera_M = Transform(extrinsic.rotation,
                             np.array([0, 0, 1])).as_matrix()  # 旋转到光轴方向 然后沿着 z 前进 1 (不用归一化了)
        direction_vector = np.linalg.inv(camera_M)[:3, 3]  # 前进之后在世界坐标下的位置 or 方向向量

        # 范围 [-0.1, 1.1] 倍的爪深, 取 depth_candidates 个
        pos_list_with_depth = []
        eps = 0.1
        if len(pos_list) == 0:
            return pos_list_with_depth, pixel_list
        
        for depth in np.linspace(-eps * finger_depth, (1.0 + eps) * finger_depth, depth_candidates):
            pos_list_with_depth.extend(pos_list + direction_vector * depth)

        return pos_list_with_depth, pixel_list * depth_candidates  # pixel_list 直接复制以对应
    