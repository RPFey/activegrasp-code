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
import open3d as o3d

import tf2_ros as tf2
from scipy.spatial.transform import Rotation as sciR
# from kortex_driver.srv import DoSensorFocusActionRequest, DoSensorFocusAction

import threading
import rospy
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import gaussian_splatting_py.grasp as MyGrasp
from gaussian_splatting_py.ros_util import convertNumpy2PoseStamped
from contact_graspnet_pytorch.config_utils import load_config
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest
from gaussian_splatting.srv import QueryPose, QueryPoseResponse, QueryPoseRequest
import subprocess

try:
    import romatch
    ROMATCH_INSTALL = True
except ImportError:
    ROMATCH_INSTALL = False

from gaussian_splatting_py.depth_anything_v2 import create_model, DepthAnythingV2
from gaussian_splatting_py.splatting import Runner as GSRunner
from gaussian_splatting_py.splatting import Config as GSConfig
from gaussian_splatting.gaussian_splatting_py.foundation.foundation import SAM2, DepthPredictionModule
from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.vision_utils import warp_image, learn_scale_and_offset_raw
from gaussian_splatting_py.tools.depth_check import convert_rgbd_to_pcd
from gaussian_splatting_py.vision_class import VisionClass, viewpoint

try:
    from gaussian_splatting.gaussian_splatting_py.sensor.realsense.rs_cam import RSCamera, RealSenseCamera
    RealSense_Enable = True
except ImportError:
    RealSense_Enable = False

CustomParser = namedtuple('CustomParser', ['points', 'points_rgb'])

def convertImg2Numpy(img:Image) -> np.ndarray:
    """ 
    Convert ROS Image message to numpy array
    
    Returns:
        res: TriggerResponse
            bool success
            string message
        img_np: np.ndarray
            (H, W, C) numpy array
    """
    res = TriggerResponse()
    res.success = True
    
    # convert to numpy array
    # process the image
    height = img.height
    width = img.width
    encoding = img.encoding

    if encoding in ["rgb8", "8UC3"]:
        img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 3))
    elif encoding == "32FC1":
        img_np = np.frombuffer(img.data, dtype=np.float32).reshape((height, width, 1))
    elif encoding == "16UC1":
        img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 1))
    elif encoding == "rgb16":
        img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 3))
    elif encoding == "rgba8":
        img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 4))
    elif encoding == "rgba16":
        img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 4))
    elif encoding == "mono8":
        img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 1))
    elif encoding == "mono16":
        img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 1))
    else:
        rospy.logerr("Unsupported image encoding: {}".format(encoding))
        res.success = False
        res.message = "Error Code 2: Unsupported image encoding: {}".format(encoding)

    return res, img_np.copy()

class VisionNode(VisionClass):
    def __init__(self) -> None:
        super().__init__()
        rospy.init_node("vision_node")

        # fetch parameter
        self.CAMERA_TOPC = rospy.get_param("~image_topic", "/image")
        self.DEPTH_TOPIC = rospy.get_param("~depth_topic", "/image")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera_info")
        self.depth_info_topic = rospy.get_param("~depth_info_topic", "/depth_info")
        self.save_data = rospy.get_param("~save_data", "False")
        self.save_data_dir = rospy.get_param("~save_data_dir", "/home/user/Documents/data")
        self.touch_config = rospy.get_param("~touch_config", "config/digit.yaml")
        self.grasp_model_path = rospy.get_param("~grasp_model_path", "")
        self.view_selection_method = rospy.get_param("~view_selection_method", "fisher")
        self.data_base_dir = osp.join(self.save_data_dir, self.date_str)
        rospy.loginfo("Camera Topic: {}".format(self.CAMERA_TOPC))
        rospy.loginfo("Depth Topic: {}".format(self.DEPTH_TOPIC))
        rospy.loginfo("Camera Info Topic: {}".format(self.cam_info_topic))
        rospy.loginfo("Depth Info Topic: {}".format(self.depth_info_topic))
        rospy.loginfo("Save Data: {}".format(self.save_data))
        rospy.loginfo("Save Data Dir: {}".format(self.save_data_dir))
    
        # try to open the camera
        global RealSense_Enable
        if RealSense_Enable:
            # try:
            self.rs_cam = RealSenseCamera()
            rospy.loginfo("RealSense Camera Initialized")
            # except Exception as e:
            #     rospy.logerr("Failed to initialize RealSense Camera")
            #     RealSense_Enable = False

        # wait for image topic
        rospy.loginfo("Waiting for camera topic {}".format(self.CAMERA_TOPC))  
        rospy.wait_for_message(self.CAMERA_TOPC, Image)
        rospy.loginfo("Waiting for depth topic {}".format(self.DEPTH_TOPIC))
        rospy.wait_for_message(self.DEPTH_TOPIC, Image)
        rospy.loginfo("Waiting for camera info topic {}".format(self.cam_info_topic))
        rospy.wait_for_message(self.cam_info_topic, CameraInfo)
        rospy.loginfo("Waiting for depth info topic {}".format(self.depth_info_topic))
        rospy.wait_for_message(self.depth_info_topic, CameraInfo)
        rospy.loginfo("Camera topic is available")

        # check if digit is available
        pub_topics = rospy.get_published_topics()
        self.DIGIT_AVAILABLE = True
        for t in pub_topics:
            if 'digit' in t[0]:
                self.DIGIT_AVAILABLE = True
                rospy.loginfo("Digit is available")
                break

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        
        # add service
        rospy.loginfo("Adding services")
        
        self.addview_srv = rospy.Service("add_view", Trigger, self.addVisionCb)
        self.addview_srv = rospy.Service("add_touch", Trigger, self.addTouchCb)
        self.showTraj_srv = rospy.Service("show_traj", Trigger, self.showTrajCb)
        self.nbv_srv = rospy.Service("next_best_view", NBV, self.NextBestView)
        self.savemodel_srv = rospy.Service("save_model", Trigger, self.saveModelCb)
        self.grasppose_pred_srv = rospy.Service("grasp_pose", QueryPose, self.queryposeCb)
        self.get_gs_data_dir_srv = rospy.Service("get_gs_data_dir", Trigger, self.getGSDataDirCb)

        # TODO initialize the radiance field
        rospy.loginfo("Vision Node Initialized")

    def showTrajCb(self, req:TriggerRequest) -> TriggerResponse:
        res = TriggerResponse()
        res.success = True

        video_path = self.gs_trainer.render_traj(0)
        subprocess.Popen(["python3.11", "-m", 
                        "gaussian_splatting_py.video_player",
                         video_path])

        subprocess.Popen(["python3.11", "-m", 
                        "gaussian_splatting_py.cam_viewer",
                         self.data_base_dir])

        return res

    def getGSDataDirCb(self, req:TriggerRequest) -> TriggerResponse:
        """ Get the data directory """
        res = TriggerResponse()
        res.success = True
        res.message = self.data_base_dir
        return res

    def convertPose2Numpy(pose:Union[PoseStamped, Pose]) -> np.ndarray:
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        c2w = np.eye(4)
        quat = sciR.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        return c2w
    
    def convertTransform2Numpy(transform:Union[TransformStamped, Transform]) -> np.ndarray:
        if isinstance(transform, TransformStamped):
            transform = transform.transform

        c2w = np.eye(4)
        quat = sciR.from_quat([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]

        return c2w
    
    def saveModelCb(self, req) -> TriggerResponse:
        """ Save Model Callback """
        res = TriggerResponse()
        res.success = True

        if self.save_data:
            self.save_images()

        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        # save the model
        self.saveModel()

        res.message = "Success"
        return res

    def addVisionCb(self, req) -> TriggerResponse:
        """ AddVision Cb 
        Steps:
            1. Get the image
            2. Get the depth
            3. Get the pose
            4. Run the DepthAnythingV2 model
            5. Add the viewpoint to the list
            6. Start the training thread (Optional)
        
        return TriggerResponse:
            bool success
            message:
                form: Error Code X: [Error Message]
                X=1 -> Training thread is still running
                X=2 -> Unsupported image encoding
                X=3 -> Failed to lookup transform from camera to base_link
        """
        res = TriggerResponse()
        res.success = True
    
        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        # do focus
        # focus_req = DoSensorFocusActionRequest()
        # focus_req.input.focus_action = 3
        # focus_req.input.sensor = 1
        # self.focus_client(focus_req)

        rate = rospy.Rate(1)
        while True:
            joint_state:JointState = rospy.wait_for_message("/joint_state", JointState)
            vel = np.array(joint_state.velocity)
            if np.max(np.abs(vel)) < 1e-3 :
                break

            rospy.loginfo("Waiting for the camera to stabilize")
            rate.sleep()

        # pause focus
        # focus_req = DoSensorFocusActionRequest()
        # focus_req.input.focus_action = 2
        # focus_req.input.sensor = 1
        # self.focus_client(focus_req)

        # grap the image message
        img:Image = rospy.wait_for_message(self.CAMERA_TOPC, Image)
        res, color_np = convertImg2Numpy(img)
        if not res.success:
            return res

        depth:Image = rospy.wait_for_message(self.DEPTH_TOPIC, Image)
        res, depth_np = convertImg2Numpy(depth)
        if not res.success:
            return res

        # get realsense image
        if RealSense_Enable:
            rs_depth, rs_image = self.rs_cam.capture()
            rs_depth = rs_depth.astype(np.float32) / 1000.
            rs_depth[rs_depth > 20] = 0
        else:
            rs_depth, rs_image = None, None
            
        # loop up the transform from camera to base_link
        color_frame = "camera_color_frame" if rs_image is None else "realsense_color_frame"
        # color_frame = "tool_frame"
        try:
            # self.listener.waitForTransform("base_link", "camera_color_frame", rospy.Time(), rospy.Duration(1))
            transform:TransformStamped = self.tfBuffer.lookup_transform("base_link", color_frame, rospy.Time())
            c2w = VisionNode.convertTransform2Numpy(transform)
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
            
        # run the DepthAnythingV2 model
        if rs_image is None:
            # use built-in camera
            cam_info:CameraInfo = rospy.wait_for_message(self.cam_info_topic, CameraInfo)
            color_K = np.array(cam_info.K).flatten()
        else:
            color_K = self.rs_cam.intrinsic_matrix().reshape(-1)
        
        if res.success:       
            self.addVision(color_np, depth_np, rs_image, rs_depth, color_K, c2w)
            res.message = "Success"
            
        # start training
        self.train_async()
        
        return res
    
    def addTouchCb(self, req) -> TriggerResponse:
        res = TriggerResponse()
        res.success = True
    
        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        if not self.DIGIT_AVAILABLE:
            res.success = False
            res.message = "Error Code 2: Digit is not available"
            return res
        
        touch = rospy.wait_for_message("/digit_node/digit_color", Image)
        res, touch_np = convertImg2Numpy(touch)

        depth = rospy.wait_for_message("/digit_node/digit_depth", Image)
        res, touch_depth_np = convertImg2Numpy(depth)

        # loop up the transform from camera to base_link
        try:
            # self.listener.waitForTransform("base_link", "camera_color_frame", rospy.Time(), rospy.Duration(1))
            transform:TransformStamped = self.tfBuffer.lookup_transform("base_link", "digit_right", rospy.Time())
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"

        touch_pose = VisionNode.convertTransform2Numpy(transform)

        # store touch
        self.views.append(viewpoint(c2w=touch_pose))
        self.views[-1].touch_frame = touch_np
        self.views[-1].touch_depth = touch_depth_np

        return res

    def NextBestView(self, req:NBVRequest) -> NBVResponse:
        """ Next Best View Service Callback """
        poses = req.poses
        res = NBVResponse()

        if len(poses) == 0:
            res.success = True
            res.message = "Success"
            return None

        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        # convert to numpy array
        poses_np = np.array([VisionNode.convertPose2Numpy(pose) for pose in poses]) # (N, 4, 4)
        scores = self.EvaluatePoses(poses_np)

        # return response
        res.success = True
        res.message = "Success"
        res.scores = list(scores)
        return res

    def saveModel(self):
        """ Save the model  """
        
        pass

    def queryposeCb(self, req):
        res = QueryPoseResponse()
        grasp_policy = req.method
        rospy.loginfo("Grasp Policy: {}".format(grasp_policy)) 
        data_base_dir = req.datadir if len(req.datadir) > 0 else self.data_base_dir  
        
        grasp_poses, meta = self.detect_grasp(grasp_policy, data_base_dir)
        
        if len(grasp_poses) > 0:
            for s, p in zip(meta["scores"], grasp_poses): 
                res.poses.append(convertNumpy2PoseStamped(p))
                res.scores.append(s)
            res.message = "Success"  
        else:
            res.message = "No Grasp Found"
        res.success = True
        torch.cuda.empty_cache()

        return res        

if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()