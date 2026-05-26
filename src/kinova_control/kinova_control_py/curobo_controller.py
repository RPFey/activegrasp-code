# there is weird import error, I decide to import this module first and everything works
import sksparse.cholmod as skch
import sys
from typing import Optional

import os
import numpy as np
import torch
import cv2
from pyquaternion import Quaternion
import open3d as o3d
import os.path as osp
from collections import namedtuple
import threading
from enum import Enum
import shutil
import time
import random
import pickle
from scipy.spatial.transform import Rotation as SciR
from tqdm import tqdm
import json
import logging

from gaussian_splatting_py.vision_class import VisionClass
from gaussian_splatting_py.foundation.sam2.frames_sam2 import run_video_prediction, SAM2_POINTS_FILENAME
from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.base import extract_object_pc

from kinova_control_py.status import *
from kinova_control_py.gripper_model import Gripper
from kinova_control_py.pose_util import calcAngDiff, convert_pose_to_pos_quat
from kinova_control_py.planner import CuroboPlanner
from kinova_control_py.tsdf_integration import O3DTsdfIntegration
from kinova_control_py.run_conformal import GraspResult

from franka_env.mp_wrapper import FrankaClutter

OBJECT_CENTER = np.array([0.5, 0., 0.25])
TRAJ_STRUCT = namedtuple("Trajectory", ["position", "velocity"])

def set_global_seed(seed: int = 42):
    """
    Sets the global random seed for numpy, Python's random module, and PyTorch.
    Ensures reproducibility across multiple runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed) # For hash-based operations
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multiple GPUs

def visualize_segmentation(rgb, seg):
    # generate random color for each catorgory
    colors = np.random.randint(0, 255, (np.max(seg)+1, 3))
    mask = np.zeros_like(rgb)
    for i in range(np.max(seg)+1):
        mask[seg == i] = colors[i]
    
        # put text for each category
        # compute the centroid
        region_x = np.where(seg == i)[1]
        region_y = np.where(seg == i)[0]
        
        if len(region_x) == 0 or len(region_y) == 0:
            continue
        
        center_x = region_x.mean().astype(int)
        center_y = region_y.mean().astype(int)
        
        if center_x < 0 or center_y < 0 \
            or center_x >= mask.shape[1] or center_y >= mask.shape[0]:
            continue

        mask = cv2.putText(mask, str(i), (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return mask       

class CuroboController:
    def __init__(self, robot_type:str, planner:str, 
                    grasp_model:str = "contact_graspnet", seed = 42):
        self.robot_type = robot_type
        if planner == "curobo":
            self.planner = CuroboPlanner(robot_type)
            
        self.logger = logging.getLogger("curobo")
        self.logger.setLevel(logging.DEBUG)
        
        self.logger_fn = self.logger.debug
        self.logger_info = self.logger.info
        self.vision = None
        self.env = None
        self.grasp_model = grasp_model
        self.seed = seed
            
    def wait_for_stabalize(self, velocity_cb, max_vel = 1e-3, time_threshold = 30):
        """ Wait for the arm to stabilize """
        start = time.time()
        while time.time() - start < time_threshold:
            vel = velocity_cb() 
            if np.max(np.abs(vel)) < max_vel:
                break
            
            self.logger_fn("Waiting for the arm to stabilize")
            time.sleep(1)
            
    def get_joint_state(self):
        """ Get the current joint state """
        state = self.env.get_joint_data()
        return state[0, :7]
    
    def get_joint_velocity(self):
        """ Get the current joint velocity """
        state = self.env.get_joint_data()
        return state[1]
    
    def close_gripper(self):
        self.env.close_gripper()
        time.sleep(3)
    
    def traj_pub_command(self, position, velocity):
        self.env.set_joint_input(position)
        time.sleep(0.03)
        
    def get_ee_pose(self):
        """ Get the current end effector pose """
        return self.env.get_ee_pose()
    
    def get_cam_pose(self):
        """ Get the current camera pose """
        return self.env.get_camera_pose()
    
    def get_finger_position(self):
        """ Get the current finger position """
        self.wait_for_stabalize(self.get_joint_velocity)
        state = self.env.get_joint_data()
        return state[0, -2:]
    
    def get_cam2hand(self):
        """ Get the camera to hand transformation """
        self.wait_for_stabalize(self.get_joint_velocity)
        ee_pose = self.get_ee_pose()
        cam_pose = self.get_cam_pose()
        return np.linalg.inv(ee_pose) @ cam_pose
        
    def add_view(self):
        self.wait_for_stabalize(self.get_joint_velocity)
        c2w = self.env.get_camera_pose()
        K = self.env.get_camera_intrinsic()
        if self.target_index <= 0:
            color, depth = self.env.get_image()
            seg = None
        else:
            color, depth, seg = self.env.get_image(return_seg=True)
            seg = (seg == self.target_index).astype(np.uint8)
        self.vision.addVision(color, depth, None, None, seg, K, c2w)
        
    def call_nbv(self, poses):
        poses = np.stack(poses, axis=0)
        return self.vision.EvaluatePoses(poses, self.grasp_model)
        
    def save_model(self):
        self.vision.save_images()
        
    def predict_grasp(self, gaussian_splatting_data_dir):
        grasps, meta = self.vision.detect_grasp(self.grasp_model, gaussian_splatting_data_dir)
        
        if len(grasps) == 0:
            return [], []
        
        return np.array(grasps), np.array(meta["scores"])
    
    def generate_poses(self, num_poses:int = 1, center = OBJECT_CENTER):
        """ Generate Poses by the pose generator, default is sphere sampling.
        The logic is:
            1. use the pose generator to generate a pose
            2. use KDL to do IK
            2. MoveIt to plan the trajectory.
        Args:
            num_poses: Number of Poses to Generate
        Return:
            candidate_joints List[np.ndarray]: List of candidate joints
            candidate_poses List[PoseStamped]: List of candidate poses
        """
        # query eye to hand
        cam2hand = self.get_cam2hand()
        
        candidate_cam_poses = []
        candidate_ee_poses = []
        
        trials = 0
        current_state = self.get_joint_state()
        while trials < num_poses * 10 and len(candidate_cam_poses) < num_poses:
            # sample a pose and use KDL to do IK
            pose = self.planner.sampleInSphere(center, 
                                               0.3, 0.5,
                                               min_theta=np.pi / 8, max_theta= 15 * np.pi / 8)

            cam_pose = np.eye(4)
            cam_pose[:3, 3] = pose[:3]
            cam_pose[:3, :3] = SciR.from_quat(pose[3:]).as_matrix()
            hand_pose = cam_pose @ np.linalg.inv(cam2hand)
            success, _ = self.planner.plan(hand_pose, current_state)
            
            if success:
                self.logger_fn("Sample one candidate pose")
                candidate_cam_poses.append(cam_pose)
                candidate_ee_poses.append(hand_pose)
            else:
                self.logger_fn("Fail to plan trajectory, continue sampling ...")

        return candidate_ee_poses, candidate_cam_poses
        
    def vision_phase_azimuth(self, init_view:int=4):
        """
        Vision Phase for 3D GS reconstruction
            Logic:
            1. Generate Random Poses
            2. Use FisherRF to get the next best view
            3. Reach the view
            4. Add the view to the model
        """
        success = True
        
        # generate poses
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 2)
        phi = np.linspace(np.pi / 9, np.pi / 4, 2)
        # phi = np.array([np.pi / 6])
        phis, thetas = np.meshgrid(phi, theta)
        phis = phis.flatten()
        thetas = thetas.flatten()
        radius = 0.35

        view_cnt = 0
        for i in range(len(thetas)):
            # sample a pose and use KDL to do IK
            pose = self.planner.poseOnSphere(thetas[i], phis[i], radius, OBJECT_CENTER, scalar_first=True)
            joints = self.planner.calcIK(pose)
        
            # planning phase, use MoveIt to plan the trajectory
            current_state = self.get_joint_state()
            success, traj = self.planner.plan(pose, current_state)

            # reach the best view
            if success:
                success &= self.planner.go(traj, command_cb=self.traj_pub_command)
                self.add_view()
                view_cnt += 1
            
            if view_cnt >= init_view:
                self.logger_info(f"Reach the view limit {init_view}, stop sampling ...")
                break
            
        # Save the model
        self.logger_fn("Save Model ...")
        self.save_model()
        return 
    
    def vision_phase_active(self, gs_data_dir, num_views=4):
        dataset = KinovaDataset(gs_data_dir)
        objects_pcs = extract_object_pc(dataset)
        if 1 not in objects_pcs:
            raise ValueError("Fail to extract object point cloud")  
        object_pc = objects_pcs[1]
        object_center = np.mean(object_pc, axis=0)
        
        # update the observations
        self.planner.fuse_observation(gs_data_dir)
        
        trial = 0
        images_taken = 0
        while trial < 8 and images_taken < num_views:
            # TODO Change number of poses
            candidate_ee_poses, poses = self.generate_poses(192, object_center)
            scores = self.call_nbv(poses)
            
            if np.max(scores) == np.min(scores):
                self.logger_fn("No Valid Next Best View")
                trial += 1
                continue
            
            best_index = np.argmax(scores)
            
            best_pose = candidate_ee_poses[best_index]
            current_state = self.get_joint_state()
            success, traj = self.planner.plan(best_pose, current_state)
            self.planner.go(traj, command_cb=self.traj_pub_command)
            
            self.add_view()
            # propogate SAM2
            # run_video_prediction(gs_data_dir, category=self.target_object)
            trial += 1
            images_taken += 1
            self.logger_info(f"Trail: {trial}, Image Taken: {images_taken} / {num_views}, Score: {scores[best_index]}")
            
        # Save the model
        self.logger_fn("Save Model ...")
        self.save_model()
    
        return True
    
    def execute_grasp(self, result):
        # Gripper.visualize_grasp_poses([pre_grasp_poses[traj_idx]], [world_mesh])
        grasp_traj = result.grasp_interpolated_trajectory
        self.planner.go(grasp_traj, command_cb=self.traj_pub_command)
        self.logger_info("Approach ...")
        self.wait_for_stabalize(self.get_joint_velocity)
        
        # Close Gripper
        self.close_gripper()
        self.logger_info("wait for finger ...")
        self.wait_for_stabalize(self.get_joint_velocity)

        finger = self.get_finger_position()
        if abs(max(finger)) > 1e-3:
            self.logger_info("Target in Gripper")
            retract_traj = result.retract_interpolated_trajectory
            self.planner.go(retract_traj, command_cb=self.traj_pub_command)
            self.logger_info("retracting ...")
            self.wait_for_stabalize(self.get_joint_velocity)
            finger = self.get_finger_position()
            if abs(max(finger)) > 1e-3:
                self.logger_info("STATUS: GraspResult.SUCCESS")
                status = GraspResult.SUCCESS
            else:
                self.logger_info("STATUS: GraspResult.DROP")
                status = GraspResult.DROP
        
        else:
            self.logger_info("STATUS: GraspResult.BAD_POSE")
            status = GraspResult.BAD_POSE

        return status

    def grasp_phase_cu(self, gaussian_splatting_data_dir):
        """ 
        Grasp Phase, using the curobo planner
        At this point, we have collected all (8) views. We now continue to the Grasp phase.
            1. Query the vision node for grasp poses.
            2. Use planner to plan trajs to pre-grasps
            3. Use linear interpolation to move between pre-grasps and grasp poses.
            4. move to home poses.
        """
        grasp_poses, scores = self.predict_grasp(gaussian_splatting_data_dir)
        score = 0. if len(scores) == 0 else scores[0]
        select_grasp = None
        
        # get the best grasp pose
        if len(grasp_poses) == 0:
            self.logger_info("Fail to Predict Grasp Pose")
            status = GraspResult.NO_VALID_GRASP
        else:
            logging.info(f"Grasp Poses: {grasp_poses}, starting grasping")
            # robot transform
            # disable collision checks for finger links since they are in contact with the object
            self.planner.toggle_collision_for_grasp()
            if self.planner.robot_type == "franka":
                t = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0, 0., 1., 0.], [0., 0., 0., 1.]])
                grasp_poses = np.stack([pose @ t for pose in grasp_poses], axis=0)
            
            # augment the grasp poses by flipping x, y axis 
            augment = np.eye(4)
            augment[[0, 1], [0, 1]] = -1
            augment_poses = np.stack([pose @ augment for pose in grasp_poses], axis=0)
            augment_scores = scores.copy()
            grasp_poses = np.concatenate([grasp_poses[:, None, :, :], augment_poses[:, None, :, :]], axis=1).reshape(-1, 4, 4)
            scores = np.concatenate([scores[None, :], augment_scores[None, :]], axis=0).reshape(-1)
            
            self.planner.fuse_observation(gaussian_splatting_data_dir)
            logging.info(f"observation fused.")
            
            # use IK to filter out the infeasible grasp poses
            success, _ = self.planner.calcIK(grasp_poses)
            success = success.cpu().numpy().reshape(-1)
            if not np.any(success):
                self.logger_info("After IK Check, No Valid Grasp Pose")
                status = GraspResult.IK_FAIL
            else:
                # sort the grasp poses by score
                grasp_poses = grasp_poses[np.argsort(scores)[::-1]]
                scores = scores[np.argsort(scores)[::-1]]

                batch_size = 8
                batch_num = int(np.ceil(len(grasp_poses) / batch_size))

                for i in range(batch_num):
                    low = i * batch_size
                    high = min((i + 1) * batch_size, len(grasp_poses))
                    
                    success, result = self.planner.plan_grasp(grasp_poses[low:high], self.get_joint_state())                
                    if success:
                        # tsdf_fusion = O3DTsdfIntegration()
                        # tsdf_fusion.build_from_data_folder(gaussian_splatting_data_dir)
                        # world_mesh = tsdf_fusion.get_mesh()
                        # Gripper.visualize_grasp_poses([grasp_poses[result.goalset_index.item()]], [world_mesh])
                        
                        grasp_traj = result.grasp_interpolated_trajectory
                        success &= self.planner.go(grasp_traj, command_cb=self.traj_pub_command)
                        self.logger_info("Approach ...")
                        score = scores[result.goalset_index.item()]
                        select_grasp = grasp_poses[result.goalset_index.item()]
                        self.logger_info("Select Pose with Score: {:.4f}".format(score))
                        self.wait_for_stabalize(self.get_joint_velocity)
                        print("Grasp Pose: ", select_grasp)
                        print("EE_pose", self.planner.calcFK(self.get_joint_state()))
                    
                        # Close Gripper
                        self.close_gripper()
                        self.logger_info("wait for finger ...")
                        self.wait_for_stabalize(self.get_joint_velocity)

                        finger = self.get_finger_position()
                        if abs(max(finger)) > 1e-3:
                            self.logger_info("Target in Gripper")
                            retract_traj = result.retract_interpolated_trajectory
                            success &= self.planner.go(retract_traj, command_cb=self.traj_pub_command)
                            self.logger_info("retracting ...")
                            self.wait_for_stabalize(self.get_joint_velocity)
                            finger = self.get_finger_position()
                            if abs(max(finger)) > 1e-3:
                                self.logger_info("STATUS: GraspResult.SUCCESS")
                                status = GraspResult.SUCCESS
                            else:
                                self.logger_info("STATUS: GraspResult.DROP")
                                status = GraspResult.DROP
                        
                        else:
                            self.logger_info("STATUS: GraspResult.BAD_POSE")
                            status = GraspResult.BAD_POSE
                        break

                else:
                    self.logger_fn("No Feasible Grasp Pose")
                    status = GraspResult.IK_FAIL

            with open(osp.join(gaussian_splatting_data_dir, "grasp_result.pkl"), "wb") as f:
                pickle.dump({"status": status.name, "score": score, "grasp": select_grasp}, f)
            
            # Create a file with the status name
            status_file_path = osp.join(gaussian_splatting_data_dir, f"status-{status.name}")
            with open(status_file_path, "w") as status_file:
                status_file.write("")

        return status, select_grasp
    
    def grasp_phase_ap(self, gaussian_splatting_data_dir, episode, top_k = 5):
        """ 
        Grasp Phase, using the curobo planner
        At this point, we have collected all (8) views. We now continue to the Grasp phase.
            1. Query the vision node for grasp poses.
            2. Use planner to plan trajs to pre-grasps
            3. Use linear interpolation to move between pre-grasps and grasp poses.
            4. move to home poses.
        """
        grasp_poses, scores = self.predict_grasp(gaussian_splatting_data_dir)
        score = 0. if len(scores) == 0 else scores[0]
        select_grasp = None
        success_grasps = []
        
        # get the best grasp pose
        if len(grasp_poses) == 0:
            self.logger_info("Fail to Predict Grasp Pose")
            status = GraspResult.NO_VALID_GRASP
        else:
            # robot transform
            # disable collision checks for finger links since they are in contact with the object
            self.planner.toggle_collision_for_grasp()
            if self.planner.robot_type == "franka":
                t = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0, 0., 1., 0.], [0., 0., 0., 1.]])
                grasp_poses = np.stack([pose @ t for pose in grasp_poses], axis=0)
            
            # augment the grasp poses by flipping x, y axis 
            augment = np.eye(4)
            augment[[0, 1], [0, 1]] = -1
            augment_poses = np.stack([pose @ augment for pose in grasp_poses], axis=0)
            augment_scores = scores.copy()
            grasp_poses = np.concatenate([grasp_poses[:, None, :, :], augment_poses[:, None, :, :]], axis=1).reshape(-1, 4, 4)
            scores = np.concatenate([scores[None, :], augment_scores[None, :]], axis=0).reshape(-1)
            
            self.planner.fuse_observation(gaussian_splatting_data_dir)
            # use IK to filter out the infeasible grasp poses
            success, _ = self.planner.calcIK(grasp_poses)
            success = success.cpu().numpy().reshape(-1)
            if not np.any(success):
                self.logger_info("After IK Check, No Valid Grasp Pose")
                status = GraspResult.IK_FAIL
            else:
                grasp_poses = grasp_poses[success]
                scores = scores[success]
                
                # sort the grasp poses by score
                grasp_poses = grasp_poses[np.argsort(scores)[::-1]]
                scores = scores[np.argsort(scores)[::-1]]

                for i in range(len(grasp_poses)):
                    if len(success_grasps) >= top_k:
                        break

                    self.env.reset()
                    success, result = self.planner.plan_grasp(grasp_poses[[i]], self.get_joint_state())
                    if success:
                        score = scores[result.goalset_index.item()]
                        self.logger_info("Select Pose with Score: {:.4f}".format(score))
                        status = self.execute_grasp(result)

                        # check the ids in hand
                        obj_id = self.env.get_object_in_hand()
                        if status == GraspResult.SUCCESS:
                            if obj_id != episode:
                                self.logger_info("Grasp Object Mismatch, target {}, grasp {}".format(episode, obj_id))
                                status = GraspResult.DROP
                            else:
                                self.logger_info("Grasp Object Correct, target {}, grasp {}".format(episode, obj_id))
                        success_grasps.append(status)
            
            if len(success_grasps) == 0:
                self.logger_info("Grasp Status: [{}]".format(status))
            else:
                results = []
                for s in success_grasps:
                    if s == GraspResult.SUCCESS:
                        results.append(1)
                    else:
                        results.append(0)
                
                self.logger_info("Grasp Status: {}".format(success_grasps)) 
                self.logger_info("Grasp Results: {}".format(results)) 
        
        return
        
    def grasp_phase(self, gaussian_splatting_data_dir):
        """ 
        Grasp Phase
        At this point, we have collected all (8) views. We now continue to the Grasp phase.
            1. Query the vision node for grasp poses.
            2. Use planner to plan trajs to pre-grasps
            3. Use linear interpolation to move between pre-grasps and grasp poses.
            4. move to home poses.
        """
        grasp_poses, scores = self.predict_grasp(gaussian_splatting_data_dir)
        score = 0. if len(scores) == 0 else scores[0]
        
        # get the best grasp pose
        if len(grasp_poses) == 0:
            self.logger_info("Fail to Predict Grasp Pose")
            status = GraspResult.NO_VALID_GRASP
        else:
            # robot transform
            # disable collision checks for finger links since they are in contact with the object
            self.planner.toggle_collision_for_grasp()
            if self.planner.robot_type == "franka":
                t = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0, 0., 1., 0.], [0., 0., 0., 1.]])
                grasp_poses = np.stack([pose @ t for pose in grasp_poses], axis=0)
            
            # augment the grasp poses by flipping x, y axis 
            augment = np.eye(4)
            augment[[0, 0], [1, 1]] = -1
            augment_poses = np.stack([pose @ augment for pose in grasp_poses], axis=0)
            augment_scores = scores.copy()
            grasp_poses = np.concatenate([grasp_poses[:, None, :, :], augment_poses[:, None, :, :]], axis=1).reshape(-1, 4, 4)
            scores = np.concatenate([scores[None, :], augment_scores[None, :]], axis=0).reshape(-1)
            
            self.planner.fuse_observation(gaussian_splatting_data_dir)
            
            # use IK to filter out the infeasible grasp poses
            success, _ = self.planner.calcIK(grasp_poses)
            success = success.cpu().numpy().reshape(-1)
            if not np.any(success):
                self.logger_info("After IK Check, No Valid Grasp Pose")
                status = GraspResult.IK_FAIL

            else:
                grasp_poses = grasp_poses[success]
                scores = scores[success]

                # iterate through each grasp pose, take the feasible one
                def convert_to_pregrasp(grasp):
                    pregrasp_distance = -0.06
                    pre_grasp = grasp.copy()
                    z_translation = pre_grasp[:3, 2] * pregrasp_distance
                    pre_grasp[:3, 3] += z_translation
                    return pre_grasp
                
                pre_grasp_poses = []
                pre_grasp_poses = np.stack([convert_to_pregrasp(grasp) for grasp in grasp_poses], axis=0)
                if len(pre_grasp_poses) > 1 :
                    traj_idx, traj = self.planner.plan_batch(pre_grasp_poses, self.get_joint_state())
                    success = traj_idx >= 0
                else:
                    success, traj = self.planner.plan(pre_grasp_poses[0], self.get_joint_state())
                    traj_idx = 0
                
                if success:
                    # Gripper.visualize_grasp_poses([pre_grasp_poses[traj_idx]], [world_mesh])
                    success &= self.planner.go(traj, command_cb=self.traj_pub_command)
                    score = scores[traj_idx]
                    self.logger_info("Select Pose with Score: {:.4f}".format(score))
                    
                    # grasp pose
                    self.wait_for_stabalize(self.get_joint_velocity)
                    success, traj = self.planner.plan(grasp_poses[traj_idx], self.get_joint_state(), True)
                    if not success:
                        self.logger_fn("Re-plan without constraint")
                        success, traj = self.planner.plan(grasp_poses[traj_idx], self.get_joint_state())
                    
                    success &= self.planner.go(traj, command_cb=self.traj_pub_command)
                    
                    # Close Gripper
                    self.wait_for_stabalize(self.get_joint_velocity)
                    self.close_gripper()
                    
                    self.wait_for_stabalize(self.get_joint_velocity)
                    finger = self.get_finger_position()
                    if abs(max(finger)) > 1e-3:
                        self.logger_info("Target in Gripper")
                        inverse_traj = TRAJ_STRUCT(traj.position.flip(0), traj.velocity.flip(0))
                        success &= self.planner.go(inverse_traj, command_cb=self.traj_pub_command)
                        
                        self.wait_for_stabalize(self.get_joint_velocity)
                        success, traj = self.planner.plan(self.home_pose, self.get_joint_state())
                        success &= self.planner.go(traj, command_cb=self.traj_pub_command)
                        
                        self.wait_for_stabalize(self.get_joint_velocity)
                        finger = self.get_finger_position()
                        if abs(max(finger)) > 1e-3:
                            self.logger_info("STATUS: GraspResult.SUCCESS")
                            status = GraspResult.SUCCESS
                        else:
                            self.logger_info("STATUS: GraspResult.DROP")
                            status = GraspResult.DROP
                    
                    else:
                        self.logger_info("STATUS: GraspResult.BAD_POSE")
                        status = GraspResult.BAD_POSE

                else:
                    self.logger_fn("No Feasible Grasp Pose")
                    status = GraspResult.IK_FAIL

        with open(osp.join(gaussian_splatting_data_dir, "grasp_result.pkl"), "wb") as f:
                        pickle.dump({"status": status.name, "score": score}, f)

        return status
    
    def run_episode(self, active_method, episode_root, seed, episode,
                        data_root = "/home/leiboshu", active_view = 4, viz=False, gs_data: Optional[str]=None, 
                        H_lambda:float = 1e-6, init_view:int=4):
        """ Run the episode """
        self.seed = seed
        self.target_index = episode

        self.vision = VisionClass(
                save_data_dir=data_root,
                view_selection_method=active_method,
                viz = viz,
                sync_training = True,
                suffix = "s{}-ep{}".format(seed, episode),
                gs_data=gs_data,
                H_lambda=H_lambda,
                logger_fn=self.logger_info)
        
        # write logging to file
        os.makedirs(self.vision.data_base_dir, exist_ok=True)
        fh = logging.FileHandler(osp.join(self.vision.data_base_dir, "curobo.log"))
        fh.setLevel(logging.DEBUG)
        self.logger_info("Episode: {}".format(episode))
        self.logger.addHandler(fh)
        self.logger_info(f"Slurm task id {os.environ.get('SLURM_ARRAY_TASK_ID', '')}, job_id {os.environ.get('SLURM_JOB_ID', '')}, array count {os.environ.get('SLURM_ARRAY_TASK_COUNT', '')}")
        self.logger_info(f"init_view: {init_view}, active_view: {active_view}")
        self.logger_info(f"hostname: {os.environ.get('HOSTNAME', '')}")
        self.logger_info(f"{sys.argv=}")
        self.logger_info(f"env: {os.environ}")

        sdf_file = os.path.join(episode_root, 'sdf', f'seed{seed}.txt')
        if os.path.exists(sdf_file):
            self.logger_info("SDF File: {}".format(sdf_file))
            load_from_list = False
        else:
            self.logger_info("SDF File Not Found, Create new seed ...")
            sdf_file = ""
            load_from_list  = True
            
        self.env = FrankaClutter(sdf_file, load_from_list, seed=seed, gui=viz, logdir=self.vision.data_base_dir)
        
        # wait to stabilize for loading from pretrained dir
        self.wait_for_stabalize(self.get_joint_velocity)
        self.home_pose = self.planner.calcFK(self.get_joint_state())
        
        if gs_data is None:
            gs_data = self.vision.data_base_dir
            # add view at the first frame
            self.add_view()
            color, depth, seg = self.env.get_image(return_seg=True)
            mask = visualize_segmentation(color, seg)
            cv2.imwrite(osp.join(gs_data, f"target_{episode}.png"), mask)
            
            # fixed views
            self.vision_phase_azimuth(init_view=init_view)
        else:
            self.vision.load(gs_data)
        
        # self.vision_phase_active(self.vision.data_base_dir, active_view)
        try:
            self.vision_phase_active(self.vision.data_base_dir, active_view)
        except ValueError as e:
            status = GraspResult.NO_VALID_GRASP
            self.logger_info("Grasp Phase done, status: {}, terminating...".format(status))
            self.env.end()
            del self.env
            return status
        torch.cuda.empty_cache()
        
        self.logger_info("Vision Phase done, starting Grasp Phase ...")
        status, grasp = self.grasp_phase_cu(gs_data)
        obj_id = self.env.get_object_in_hand()
        
        # check the ids in hand
        if status == GraspResult.SUCCESS:
            if episode not in obj_id:
                self.logger_info("Grasp Object Mismatch, target {}, grasp {}".format(episode, obj_id))
                status = GraspResult.DROP
            else:
                self.logger_info("Grasp Object Correct, target {}, grasp {}".format(episode, obj_id))
            
        self.logger_info("Grasp Phase done, status: {}, terminating...".format(status))
        self.env.end()
        del self.env
        return status
    
    def run_episode_ap(self, active_method, episode_root, seed, episode,
                        data_root = "/home/leiboshu", active_view = 4, viz=False, 
                        gs_data: Optional[str]=None, H_lambda:float = 1e-6, 
                        init_view:int=4, top_k = 5):
        """ Run the episode """
        self.seed = seed
        self.target_index = episode

        self.vision = VisionClass(
                save_data_dir=data_root,
                view_selection_method=active_method,
                viz = viz,
                sync_training = True,
                suffix = "s{}-ep{}".format(seed, episode),
                gs_data=gs_data)
        
        # write logging to file
        os.makedirs(self.vision.data_base_dir, exist_ok=True)
        fh = logging.FileHandler(osp.join(self.vision.data_base_dir, "curobo.log"))
        fh.setLevel(logging.DEBUG)
        self.logger_info("Episode: {}".format(episode))
        self.logger.addHandler(fh)

        sdf_file = os.path.join(episode_root, 'sdf', f'seed{seed}.txt')
        if os.path.exists(sdf_file):
            self.logger_info("SDF File: {}".format(sdf_file))
            load_from_list = False
        else:
            self.logger_info("SDF File Not Found, Create new seed ...")
            sdf_file = ""
            load_from_list  = True
            
        self.env = FrankaClutter(sdf_file, load_from_list, seed=seed, gui=viz, logdir=self.vision.data_base_dir)
        
        # wait to stabilize for loading from pretrained dir
        self.wait_for_stabalize(self.get_joint_velocity)
        self.home_pose = self.planner.calcFK(self.get_joint_state())
        
        if gs_data is None:
            gs_data = self.vision.data_base_dir
            # add view at the first frame
            self.add_view()
            color, depth, seg = self.env.get_image(return_seg=True)
            mask = visualize_segmentation(color, seg)
            cv2.imwrite(osp.join(gs_data, f"target_{episode}.png"), mask)
            
            # fixed views
            self.vision_phase_azimuth(init_view=init_view)
        else:
            self.vision.load(gs_data)
        
        self.vision_phase_active(self.vision.data_base_dir, active_view)
        self.logger_info("Vision Phase done, starting Grasp Phase ...")

        self.grasp_phase_ap(gs_data, episode, top_k)
        self.env.end()
        del self.env
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", type=str, default="franka")
    parser.add_argument("--planner", type=str, default="curobo")
    parser.add_argument("--grasp_model", type=str, default="contact_graspnet")
    parser.add_argument("--active_method", type=str, default="Breyer")
    parser.add_argument("--active_view", type=int, default=4)
    parser.add_argument("--data_root", type=str, default="/home/leiboshu")
    parser.add_argument("--ep_root", type=str, default=None)
    parser.add_argument("--urdf_list", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=-1, 
                        help=" the id for objects in the current scene, start from 1 - no. objects ")
    parser.add_argument("--viz", action="store_true", default=False)
    parser.add_argument("--gs_data", type=str, default=None)
    parser.add_argument("--ap", action="store_true", default=False, help="measure the ap in each experiment")
    parser.add_argument("--top_k", type=int, default=5, help="top k grasps to test")
    parser.add_argument("--H_lambda", type=float, default=1e-6)
    parser.add_argument("--init_view", type=int, default=4)
    opt = parser.parse_args()
    
    # set random seed
    set_global_seed(opt.seed)
    
    planner = CuroboController(opt.robot_type, opt.planner, opt.grasp_model, seed=opt.seed)
    if opt.ap:
        planner.run_episode_ap(opt.active_method, opt.ep_root, opt.seed, opt.episode, opt.data_root, opt.active_view, opt.viz, 
                        gs_data=opt.gs_data, H_lambda=opt.H_lambda, init_view=opt.init_view, top_k=opt.top_k)
    else:
        planner.run_episode(opt.active_method, opt.ep_root, opt.seed, opt.episode, opt.data_root, opt.active_view, opt.viz, 
                            gs_data=opt.gs_data, H_lambda=opt.H_lambda, init_view=opt.init_view)

