#!/usr/bin/env python3

import argparse
import copy
import datetime
import json
import logging
import os
import os.path as osp
import subprocess
import threading
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Union, Callable

import cv2
import gaussian_splatting_py.grasp as MyGrasp
import matplotlib
import matplotlib.image
import numpy as np
import open3d as o3d
import torch
import tqdm
from contact_graspnet_pytorch.config_utils import load_config
from scipy.spatial.transform import Rotation as sciR

try:
    import romatch
    ROMATCH_INSTALL = True
except ImportError:
    ROMATCH_INSTALL = False

from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.foundation.depth_anything_v2 import (
    DepthAnythingV2, create_model)
from gaussian_splatting_py.foundation.foundation import (SAM2,
                                                         DepthPredictionModule)
from gaussian_splatting_py.splatting import Config as GSConfig
from gaussian_splatting_py.splatting import Runner as GSRunner
from gaussian_splatting_py.tools.depth_check import convert_rgbd_to_pcd
from gaussian_splatting_py.vision_utils import (learn_scale_and_offset_raw,
                                                warp_image)

try:
    from gaussian_splatting_py.sensor.realsense.rs_cam import (RealSenseCamera,
                                                               RSCamera)
    RealSense_Enable = True
except ImportError:
    RealSense_Enable = False

CustomParser = namedtuple('CustomParser', ['points', 'points_rgb'])
dirname = osp.dirname(osp.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class viewpoint:
    # 4x4 c2w pose
    c2w: np.ndarray
    image: np.ndarray
    monodepth: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    
    rs_image: np.ndarray
    rs_depth: np.ndarray
    rs_pose: np.ndarray

    touch_depth: np.ndarray
    touch_frame: np.ndarray

    def __init__(self, c2w=None, image=None, monodepth=None, 
                    depth=None, mask=None,
                    rs_pose=None, rs_image=None, rs_depth=None) -> None:
        self.c2w = c2w
        self.image = image
        self.monodepth = monodepth
        self.depth = depth
        self.mask = mask
        self.rs_image = rs_image
        self.rs_depth = rs_depth
        self.rs_pose = rs_pose

        self.touch_depth = None
        self.touch_frame = None
        
class VisionClass:
    def __init__(self, 
                    save_data_dir:dir="/root/data",
                    view_selection_method:str="Breyer",
                    viz:bool=False,
                    sync_training:bool=False,
                    suffix="",
                    verbose=False,
                    gs_data:Optional[str]=None,
                    H_lambda:float=1e-6,
                    logger_fn:Callable=print):
        """
        Args:
            save_data_dir: str, the directory to save the captured images
            view_selection_method: str, the method to select the next view
                    "random", "fisher", "Breyer"
            viz: bool, whether to visualize the training process
            seed: int, the random seed
        """
        # training thread
        self.gs_trainer = None
        self.gs_att_mask = None
        self.training_thread = threading.Thread(target=self._training_loop)
        self.touch_config = "config/digit.yaml"
        self.view_selection_method = view_selection_method
        self.viz = viz
        self.sync_training = sync_training
        self.verbose = verbose

        # Load the DepthAnythingV2 model
        self.dpt_model = None
        # dp_encoder = rospy.get_param("~depth_encoder", "vitl")
        # dp_model_path = rospy.get_param("~depth_model_path", None)
        # try:
        #     self.dpt_model = DepthPredictionModule(dp_encoder, dp_model_path)
        #     rospy.loginfo("DepthAnythingV2 Model Loaded")
        # except Exception as e:
        #     rospy.logerr("Failed to load DepthAnythingV2 Model")
        #     self.dpt_model = None
        # self.sam = SAM2()
        
        self.save_data_dir = save_data_dir
        now = datetime.datetime.now()
        # add seed option
        if gs_data is not None: # fast loading for debugging
            self.data_base_dir = gs_data
            self.date_str = osp.basename(gs_data)
        else:
            self.date_str = now.strftime("%Y-%m-%d-%H-%M-%S") + '-' + suffix
            self.data_base_dir = osp.join(self.save_data_dir, self.date_str)
        
        # store images in (H, W, 3) [0 - 255]
        self.views:List[viewpoint] = []
        self.H_lambda = H_lambda
        self.logger_fn:Callable = logger_fn
        self.logger_fn(f"H_lambda: {self.H_lambda}")  
        
    def addVision(self, color_np, depth_np, rs_image=None, rs_depth=None, mask = None, 
                            color_K=np.eye(3), c2w=np.eye(4)):
        if len(self.views) == 0:
            image_height = rs_image.shape[0] if rs_image is not None else color_np.shape[0]
            image_width = rs_image.shape[1] if rs_image is not None else color_np.shape[1]
        
            self.json_txt = {
                "w": image_width, "h": image_height,
                "fl_x": color_K[0, 0].item(), "fl_y": color_K[1, 1].item(),
                "cx": color_K[0, 2].item(), "cy": color_K[1, 2].item()
            }
        
        # convert to meters
        if depth_np.dtype == np.uint16:
            depth_np = depth_np.astype(np.float32) / 1000. 
        depth_np[depth_np > 20] = 0
                
        monodepth = None
        if self.dpt_model is not None:
            color_input = rs_image if rs_image is not None else color_np
            depth_input = rs_depth if rs_depth is not None else depth_np
            
            print("Running DepthAnythingV2 model")
            mde_inverse, mde_depth = self.dpt_model.predict_depth(color_input)

            # realsense has already aligned depth to color
            warp_depth, _ = self.dpt_model.align_depth(depth_input, mde_depth, color_K, color_np, warp_depth_frame = rs_image is None)
            _, _, monodepth = self.sam.generate(color_input, mde_depth, warp_depth)

        self.views.append(viewpoint(c2w=c2w, image=color_np, depth=depth_np, mask=mask, monodepth=monodepth, rs_image=rs_image, rs_depth=rs_depth))
        
        # c, d, pcd = convert_rgbd_to_pcd(color_np, depth_np, color_K.reshape(3, 3), np.eye(4))
        # o3d.visualization.draw_geometries([pcd])
        
        if self.sync_training:
            self.train_sync()
        else:
            self.train_async()
        
    def train_async(self):
        """ Train the 3dgs asynchronously """
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
        
    def train_sync(self):
        """ Train the 3dgs synchronously """
        self._training_loop()
        
    def save_images(self):
        """ Save the captured images as a NeRF Synthetic Dataset format """
        # get camera info
        # get the date format in Year-Month-Day-Hour-Minute-Second
        assert len(self.views) > 0, "No views to save"

        data_base_dir = self.data_base_dir 
        print("Save Data Dir: ", data_base_dir)
        os.makedirs(data_base_dir, exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "depths"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "rs_images"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "rs_depths"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "monodepths"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "monodepths_viz"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "touch_frames"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "touch_depths"), exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "mask"), exist_ok=True)
        
        self.json_txt["frames"] = []

        # get colormap 
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        for img_idx, view in enumerate(self.views):            
            # save the pose
            pose = view.c2w.flatten()
            pose = [p.item() for p in pose]
            pose_list = [pose[:4], pose[4:8], pose[8:12], pose[12:]]
            pose_info = {
                "transform_matrix": pose_list,
            }

            image = view.image
            if image is not None:
                pose_info["file_path"] = osp.join("images", "{:04d}.png".format(img_idx))
                image = image[:, :, ::-1] # RGB to BGR
                cv2.imwrite(osp.join(data_base_dir, "images", "{:04d}.png".format(img_idx)), image)

            # process monodepth
            monodepth = view.monodepth
            if monodepth is not None:
                mono = (monodepth * 1000).astype(np.uint16)
                cv2.imwrite(osp.join(data_base_dir, "monodepths", "{:04d}.png".format(img_idx)), mono)
                pose_info["monodepth_path"] = osp.join("monodepths", "{:04d}.png".format(img_idx))

                # save the monodepth visualization
                monodepth_vis = (monodepth - monodepth.min()) / (monodepth.max() - monodepth.min()) * 255.
                monodepth_vis = monodepth_vis.astype(np.uint8)
                monodepth_vis = (cmap(monodepth_vis)[:, :, :3] * 255).astype(np.uint8)
                monodepth_vis_path = osp.join(data_base_dir, "monodepths_viz", "{:04d}_vis.png".format(img_idx))
                matplotlib.image.imsave(monodepth_vis_path, monodepth_vis)

            # process depth
            depth_np = view.depth
            if depth_np is not None:
                depth_np = (depth_np * 1000).astype(np.uint16)
                depth_path = osp.join(data_base_dir, "depths", "{:04d}.png".format(img_idx))
                cv2.imwrite(depth_path, depth_np)
                pose_info["depth_path"] = osp.join("depths", "{:04d}.png".format(img_idx))

            rs_image = view.rs_image
            if rs_image is not None:
                rs_image_path = osp.join(data_base_dir, "rs_images", "{:04d}.png".format(img_idx))
                cv2.imwrite(rs_image_path, rs_image)
                pose_info["rs_image_path"] = osp.join("rs_images", "{:04d}.png".format(img_idx))
            
            rs_depth = view.rs_depth
            if rs_depth is not None:
                rs_depth = (rs_depth * 1000).astype(np.uint16)
                rs_depth_path = osp.join(data_base_dir, "rs_depths", "{:04d}.png".format(img_idx))
                cv2.imwrite(rs_depth_path, rs_depth)
                pose_info["rs_depth_path"] = osp.join("rs_depths", "{:04d}.png".format(img_idx))

            # save touch
            touch_frame = view.touch_frame
            if touch_frame is not None:
                touch_frame_path = osp.join(data_base_dir, "touch_frames", "{:04d}.png".format(img_idx))
                cv2.imwrite(touch_frame_path, touch_frame)
                pose_info["touch_frame_path"] = osp.join("touch_frames", "{:04d}.png".format(img_idx))

            touch_depth = view.touch_depth
            if touch_depth is not None:
                # touch_depth = (touch_depth * 1000).astype(np.uint16)
                assert touch_depth.dtype == np.uint16, "Touch depth should have been scaled as integers"
                touch_depth_path = osp.join(data_base_dir, "touch_depths", "{:04d}.png".format(img_idx))
                cv2.imwrite(touch_depth_path, touch_depth)
                pose_info["touch_depth_path"] = osp.join("touch_depths", "{:04d}.png".format(img_idx))
                
            mask = view.mask
            if mask is not None:
                mask = mask * 255 if mask.max() == 1 else mask
                mask_path = osp.join(data_base_dir, "mask", "{:04d}.png".format(img_idx))
                cv2.imwrite(mask_path, mask)
                pose_info["mask"] = osp.join("mask", "{:04d}.png".format(img_idx))

            self.json_txt["frames"].append(pose_info)

        # dump to json file
        json_file = osp.join(data_base_dir, "transforms.json")
        with open(json_file, "w") as f:
            json.dump(self.json_txt, f)
    
    def init_gs_cfg(self, training_step:int, train_dataset:KinovaDataset) -> GSConfig:
        cfg = GSConfig()
        # cfg.init_type = "random"

        # set running steps
        cfg.max_steps = training_step
        # cfg.init_scale = 0.005
        cfg.init_scale = 0.05
        cfg.init_type = "romatch"
        cfg.depth_loss = len(train_dataset.depths) > 0
        cfg.result_dir = osp.join(self.data_base_dir, "gs")

        # Extra config after introducing semnatics
        cfg.semantic_mode = True
        cfg.sh_degree = 0
        cfg.init_opa = 0.9
        cfg.isotropic = True

        if os.environ.get("SLURM_ARRAY_TASK_ID", None) is not None:
            cfg.disable_viewer = True # disable viewer for slurm jobs

        return cfg
            
    def _training_loop(self):
        """ Train the Radiance Field """
        if len(self.views) < 2:
            print("Not enough views to train the model")
            return
        
        interval = 1
        
        # save images and then load to dataset 
        self.save_images()
        train_dataset = KinovaDataset(self.data_base_dir, touch_cfg=self.touch_config)
        training_step = 1000
 
        # initialize the GSTrainer if first called
        if self.gs_trainer is None:

            cfg = self.init_gs_cfg(training_step, train_dataset)
            
            # "sfm" will use the pointcloud as initialization
            if cfg.init_type == "sfm":

                raise NotImplementedError("SFM initialization is not implemented yet")
                # depth point cloud initialization
                pcd_msg:PointCloud2 = rospy.wait_for_message("/camera/depth/points", PointCloud2)
                frame_name = pcd_msg.header.frame_id
                # look up the transform from camera to base_link
                try:
                    transform:TransformStamped = self.tfBuffer.lookup_transform("base_link", frame_name, rospy.Time())
                except Exception as e:
                    rospy.logerr("Failed to lookup transform from camera to base_link")
                    return
                
                p2w = VisionNode.convertTransform2Numpy(transform)
                # check the fields
                xyz = np.frombuffer(pcd_msg.data, dtype=np.float32).copy().reshape(-1, 3)
                nan_value = np.isnan(xyz).any(axis=1)
                xyz = xyz[~nan_value]
        
                xyz = np.dot(xyz, p2w[:3, :3].T) + p2w[:3, 3]
                rgb = np.random.rand(xyz.shape[0], 3) # random initialization
                parser = CustomParser(points=xyz, points_rgb=rgb)
            else:
                parser = None

            print("Initializing the GSTrainer")
            self.gs_trainer = GSRunner(0, 0, 1, cfg, train_dataset=train_dataset, parser=parser)
        
        if len(self.views) % interval == 0:
            # update the trainset
            self.gs_trainer.trainset = train_dataset
            self.gs_trainer.cfg.max_steps += training_step * interval

            # run the training process
            self.gs_trainer.train(begin_step = self.gs_trainer.cfg.max_steps - training_step * interval,
                                    evaluate=False)
            torch.cuda.empty_cache()
        
        if self.gs_trainer.cfg.max_steps not in self.gs_trainer.cfg.save_steps:
            self.gs_trainer.save_ckpt(step=self.gs_trainer.cfg.max_steps)
        return 
    
    def EvaluatePoses(self, poses:np.ndarray, grasp_model_name:str) -> np.ndarray:
        """ Evaluate poses  """
        
        # TODO update
        if self.view_selection_method == "Fisher":
            poses = torch.from_numpy(poses).float()
            
            Ks = np.array([[self.json_txt["fl_x"], 0, self.json_txt["cx"]], 
                           [0, self.json_txt["fl_y"], self.json_txt["cy"]], 
                           [0, 0, 1]])
            
            Ks = torch.from_numpy(Ks).float()
            cam_height = self.json_txt["h"]
            cam_width = self.json_txt["w"]
            
            # train views ...
            trainviews = np.stack([view.c2w for view in self.views], axis=0)
            trainviews = torch.from_numpy(trainviews).float()
            H_train = self.gs_trainer.computeHtrain(trainviews, Ks, cam_width, cam_height)
            H_train_inv = torch.reciprocal(H_train + 1e-6)

            # compute EIG
            scores = []
            for p in poses:
                H, _ = self.gs_trainer.computeHessian(p, Ks, cam_width, cam_height)    
                if self.gs_att_mask is not None and len(self.gs_att_mask) == self.gs_trainer.splats["means"].shape[0]:
                    # apply the mask to compute EIG
                    EIG = torch.sum(H * H_train_inv * self.gs_att_mask[:, None], dim=1)
                else:
                    EIG = torch.sum(H * H_train_inv, dim=1)
                scores.append(EIG.mean().item())
            
            scores = np.array(scores)
        
        elif self.view_selection_method == "FisherGrasp":
            dataset = self._prepare_data(self.data_base_dir)
            
            # Compute H train
            poses = torch.from_numpy(poses).float()
            
            Ks = np.array([[self.json_txt["fl_x"], 0, self.json_txt["cx"]], 
                           [0, self.json_txt["fl_y"], self.json_txt["cy"]], 
                           [0, 0, 1]])
            
            Ks = torch.from_numpy(Ks).float()
            cam_height = self.json_txt["h"]
            cam_width = self.json_txt["w"]

            H_recon_train = None
            for pack in dataset:
                H, _ = self.gs_trainer.computeHessian(pack["camtoworld"].to(device), pack["K"].to(device), 
                                                      cam_width, cam_height, fisher_color=0., fisher_depth=1.)
                if H_recon_train is None:
                    H_recon_train = H
                else:
                    H_recon_train += H
            
            # Compute the Hessian from grasp model
            # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
            # FLAGS = MyGrasp.CONTACTGRASP_CONFIG("", [0.1, 1.1], True, True, False, 5)
            # policy = MyGrasp.SE3Diffusion(FLAGS)
            FLAGS = MyGrasp.CONTACTGRASP_CONFIG("", [0.1, 1.1], True, True, False, 5)
            model_params_dict = dict(se3diff="partial_grasp_dif",
                                     se3diff_scene="multiobject_scene_graspdif",
                                     se3diff_dual="multiobject_scene_graspdif_dual")
            model_name = model_params_dict[grasp_model_name]
            policy = MyGrasp.SE3Diffusion(FLAGS, model_name)
            
            # predict grasps & scores
            eta_val, J_grasp = policy.backpropagate(dataset=dataset, viz=self.viz, splats=self.gs_trainer.splats)
            H_grasp = J_grasp * J_grasp # + self.H_lambda
            
            second_term = torch.sum( H_grasp * torch.reciprocal(H_recon_train + self.H_lambda) ).item()
            self.logger_fn(f"eta_estimated first term: {eta_val:.4f} \n \
                                variance term: {second_term:.4f}")

            # if os.environ.get("H_TRAIN_POS_ONLY", None) is not None:
            #     H_train_inv = torch.reciprocal(H_train + self.H_lambda) * (H_train > self.H_lambda).float()
            # elif os.environ.get("H_TRAIN_NO_INV", None) == "True":
            #     H_train_inv = H_train
            # else:
            #     H_train_inv = torch.reciprocal(H_train + self.H_lambda) 
            
            # TODO: Optimize efficiency by computing means only
            # compute EIG
            scores = []
            for p in poses:
                H, _ = self.gs_trainer.computeHessian(p, Ks, cam_width, cam_height, fisher_color=0., fisher_depth=1.)    
                if self.gs_att_mask is not None and len(self.gs_att_mask) == self.gs_trainer.splats["means"].shape[0]:
                    # apply the mask to compute EIG
                    EIG = torch.sum(H * H_train_inv * self.gs_att_mask[:, None], dim=1)
                else:
                    EIG = torch.sum(H_grasp * ( torch.reciprocal(H_recon_train + self.H_lambda) - torch.reciprocal(H_recon_train + H + self.H_lambda)), dim=1)
                scores.append(EIG.sum().item())
            
            scores = np.array(scores)
            self.logger_fn(f"Scores mean: {scores.mean():.4f}, std: {scores.std():.4f}, max: {scores.max():.4f}, min: {scores.min():.4f}")
        
        elif self.view_selection_method == "Breyer":
            VGN_CONFIG = namedtuple("VGN_CONFIG", ["finger_depth", "model_path"])
            vgn_model = osp.join(dirname, "..", "weights/vgn_conv.pth")
            config = VGN_CONFIG(0.05, vgn_model)
            policy = MyGrasp.VGNGrasp(config)
            
            gs_data = self.data_base_dir # "/root/data/2025-01-17-16-11-07"
            scores = policy.compute_IG(poses, KinovaDataset(gs_data))
            
        elif self.view_selection_method == "ActiveNGF":
            policy = MyGrasp.ActiveNGFEstimator("active")
            
            gs_data = self.data_base_dir # "/root/data/2025-01-17-16-11-07"
            scores = policy.compute_IG(poses, KinovaDataset(gs_data))
            
        elif self.view_selection_method == "ACE":
            policy = MyGrasp.ACEGraspEstimator("ACE", 128)
            gs_data = self.data_base_dir
            scores = policy.compute_IG(poses, KinovaDataset(gs_data))
            
        elif self.view_selection_method == "random":
            scores = np.random.rand(poses.shape[0])
        else:
            raise ValueError(f"Unknown view_selection_method: {self.view_selection_method}")

        # one more based on Conformal, I think, as an ablation with pure FisherRF method.   
        scores = np.array(scores)
        
        # save viz
        np.savez(os.path.join(self.data_base_dir, f"pose_eval_{len(self.views)}.npz"), scores=scores, poses=poses)
        
        return scores
        
    def detect_grasp(self, grasp_policy:str, data_base_dir:str) -> Union[np.ndarray, dict]:
        dataset = self._prepare_data(data_base_dir)
        model_params_dict = dict(se3diff="partial_grasp_dif", 
                                se3diff_scene="multiobject_scene_graspdif", 
                                se3diff_dual="multiobject_scene_graspdif_dual")
        
        # Foundation Pose Method
        if grasp_policy == "foundationPose":
            policy = MyGrasp.FoundationPoseObjectGrasp(
                self.mesh_path, self.grasp_pose_path, self.score_model_path, self.refine_model_path
            )
            grasp_pose = policy.predict_grasp_pose(gaussian_splatting_data_dir=data_base_dir)
        
        # Contact Grasp Net Method
        elif grasp_policy == "contact_graspnet":
            import contact_graspnet_pytorch
            package_dir = osp.dirname(contact_graspnet_pytorch.__file__)
            ckpt_dir = osp.join(package_dir, "..", "checkpoints", "contact_graspnet")
            
            global_config = load_config(ckpt_dir, batch_size=5)
            
            # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
            FLAGS = MyGrasp.CONTACTGRASP_CONFIG(ckpt_dir, [0.1, 1.1], True, True, False, 5)
            policy = MyGrasp.ContactGraspNet(
                global_config, FLAGS
            )
            # predict grasps & scores
            grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=self.viz)
        
        # SE3 Diffusion Model
        elif grasp_policy in model_params_dict.keys():
            FLAGS = MyGrasp.CONTACTGRASP_CONFIG("", [0.1, 1.1], True, True, False, 5)
            model_name = model_params_dict[grasp_policy]
            policy = MyGrasp.SE3Diffusion(FLAGS, model_name, 1024)
            # predict grasps & scores
            grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=self.viz, pcs_from_splats=False)
            
        # ETH VGN method
        elif grasp_policy == "vgn":
            VGN_CONFIG = namedtuple("VGN_CONFIG", ["finger_depth", "model_path"])
            vgn_model = osp.join(dirname, "..", "weights/vgn_conv.pth")
            config = VGN_CONFIG(0.05, vgn_model)
            policy = MyGrasp.VGNGrasp(config)
            grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=self.viz)
        
        # Active NGF Method
        elif grasp_policy == "ActiveNGF":
            policy = MyGrasp.ActiveNGFEstimator("active")
            grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=self.viz)
                  
        # Ours
        elif grasp_policy == "ACE":
            policy = MyGrasp.ACEGraspEstimator("ACE", 128)
            grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=self.viz)

        else:
            raise NotImplementedError(f"Grasp Policy Not Implemented {grasp_policy}")

        return grasp_poses, meta
    
    def _prepare_data(self, data_base_dir:str, 
                        use_depth_render = False, 
                        use_color_render = False):
        """ Prepare the data for grasp detection """
        # save images and then load to dataset 
        train_dataset = KinovaDataset(data_base_dir, touch_cfg=self.touch_config)
        num_views = len(train_dataset)
        
        data_packs = []
        for view_id in range(num_views):
            data_pack = train_dataset[view_id]
            data_pack["root_dir"] = train_dataset.root_dir
            K = data_pack["K"].to(device) 
            c2w = data_pack["camtoworld"].to(device).unsqueeze(0)
            height, width = data_pack["image"].shape[:2]
            
            if use_color_render or use_depth_render:
                render = self.gs_trainer.render_at_pose(c2w, K, 
                                                        height, width, "color")
                
                if use_color_render:
                    data_pack["image"] = render["colors"]
                    
                if use_depth_render:
                    data_pack["depths"] = render["depth"]
                
            data_packs.append(data_pack)
        
        return data_packs
    
    def load(self, run_dir:str):
        """ Load from the save dir """
        train_dataset = KinovaDataset(run_dir)
        
        for view in train_dataset:
            H, W = view["image"].shape[:2]

            cur_mask = view["mask"].numpy() if view["mask"] is not None else None
            self.views.append(viewpoint(c2w=view["camtoworld"].numpy(), image=view["image"].numpy(), depth=view["depths"].reshape(H, W).numpy(), mask=cur_mask))
        
        training_step = 500
        cfg = self.init_gs_cfg(training_step, train_dataset)

        self.gs_trainer = GSRunner(0, 0, 1, cfg, train_dataset=train_dataset, parser=None)

        # load weights
        ckpts = os.listdir(osp.join(run_dir, "gs", "ckpts"))
        ckpts.sort(key=lambda x: int(x.split("_")[1]))
        load_ckpt_path = osp.join(run_dir, "gs", "ckpts", ckpts[-1])
        print(f"Loading checkpoint from {load_ckpt_path}")
        ckpt = torch.load(load_ckpt_path, map_location=device, weights_only=True)
        for k in self.gs_trainer.splats.keys():
            self.gs_trainer.splats[k].data = ckpt["splats"][k]

        self.gs_trainer.trainset = train_dataset
        self.gs_trainer.cfg.max_steps = ckpt["step"] + training_step

        # load transform json 
        json_file = osp.join(run_dir, "transforms.json")
        with open(json_file) as f:
            self.json_txt = json.load(f)
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/root/data")
    args.add_argument("--grasp_model", type=str, default="contact_graspnet")
    args.add_argument("--viz", action="store_true", default=False)
    opt = args.parse_args()
    
    vision = VisionClass("/tmp/grasp_data", viz=opt.viz)
    vision.detect_grasp(opt.grasp_model, opt.data_dir)
    