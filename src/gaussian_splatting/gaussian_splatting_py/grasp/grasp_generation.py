#!/usr/bin/env python3

import os
import datetime
import os.path as osp
import matplotlib.image
import numpy as np
import cv2
import json
from typing import List, Union
import matplotlib
from dataclasses import dataclass
from collections import namedtuple
import torch
import argparse
import copy
import open3d as o3d
import pickle as pkl

from scipy.spatial.transform import Rotation as sciR
# from kortex_driver.srv import DoSensorFocusActionRequest, DoSensorFocusAction

import threading
import gaussian_splatting_py.grasp as MyGrasp
from contact_graspnet_pytorch.config_utils import load_config
    
from franka_env.grasp_generator import ClutterRemovalSim, render_images, evaluate_grasp_pose, Label
from franka_env.btsim import Rotation, Transform, CameraIntrinsic

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/root/data")
    args.add_argument("--grasp_model", type=str, default="contact_graspnet")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument('--num_objects', type=int, default=2)
    args.add_argument("--target_index", type=int, default=2)
    args.add_argument("--viz", action="store_true", default=False)
    opt = args.parse_args()
    
    os.makedirs(opt.data_dir, exist_ok=True)
    save_dir = osp.join(opt.data_dir, opt.grasp_model, f"seed{opt.seed}-obj{opt.num_objects}-t{opt.target_index}")
    os.makedirs(save_dir, exist_ok=True)
    sim = ClutterRemovalSim("pile", gui=True, seed=opt.seed)
    sim.reset(opt.num_objects)
    sim.save_state()
    
    # render synthetic depth images
    # MAX_VIEWPOINT_COUNT = 4
    n = 8 # np.random.randint(MAX_VIEWPOINT_COUNT) + 1
    depth_imgs, extrinsics, segs = render_images(sim, n)
    
    target = opt.target_index
    assert target >= 2 and target < opt.num_objects + 2
    dataset = []
    for depth, ext, seg in zip(depth_imgs, extrinsics, segs):
        # process intrinsic
        K = np.eye(3)
        K[0, 0] = sim.camera.intrinsic.fx
        K[1, 1] = sim.camera.intrinsic.fy
        K[0, 2] = sim.camera.intrinsic.cx
        K[1, 2] = sim.camera.intrinsic.cy
        
        color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        color[..., 0] = 255
        target_mask = (seg == target).astype(np.uint8)
        color[target_mask == 1] = [0, 255, 0]
        
        dataset.append(
            {
                "root_dir": save_dir,
                "image": color,
                "depths": depth,
                "K": K,
                "camtoworld": np.linalg.inv(Transform.from_list(ext).as_matrix()),
                "mask": (seg == target).astype(np.uint8)
            }
        )
    
    if opt.grasp_model == "contact_graspnet":
        # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
        import contact_graspnet_pytorch
        package_dir = osp.dirname(contact_graspnet_pytorch.__file__)
        ckpt_dir = osp.join(package_dir, "..", "checkpoints", "contact_graspnet")
        global_config = load_config(ckpt_dir, batch_size=5)
        
        FLAGS = MyGrasp.CONTACTGRASP_CONFIG(ckpt_dir, [0.1, 1.1], True, True, False, 5)
        policy = MyGrasp.ContactGraspNet(
            global_config, FLAGS
        )
        
    elif opt.grasp_model in ["se3diff", "se3diff_ap"]:
        # "ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"
        FLAGS = MyGrasp.CONTACTGRASP_CONFIG("", [0.1, 1.1], True, True, False, 5)
        policy = MyGrasp.SE3Diffusion(FLAGS)
    
    # predict grasps & scores
    grasp_poses, meta = policy.predict_grasp_pose(dataset=dataset, viz=True)
    results = []
    
    if len(grasp_poses) > 0:
        for grasp in grasp_poses:
            execute_grasp = grasp.copy()
            execute_grasp[:3, 3] += 0.065 * execute_grasp[:3, 2]
            execute_grasp = execute_grasp @ np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            sim.restore_state()
            outcome, width = evaluate_grasp_pose(sim, execute_grasp)
            if outcome == Label.SUCCESS:
                print("Grasp success")
                results.append(1)
            else:
                results.append(0)
        
        results = np.array(results)
        
        with open(osp.join(save_dir, "results.pkl"), "wb") as f:
            pkl.dump({"grasps": grasp_poses, "results": results}, f)