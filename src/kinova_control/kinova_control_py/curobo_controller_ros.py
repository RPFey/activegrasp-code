import numpy as np
import torch
from pyquaternion import Quaternion
import open3d as o3d
import os.path as osp
from collections import namedtuple
import threading
import shutil
import time
from scipy.spatial.transform import Rotation as SciR

# ROS Srv & Msgs
import rospy 
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, Empty
from geometry_msgs.msg import PoseStamped, Pose
import tf2_ros as tf2

from gaussian_splatting_py.ros_util import convertNumpy2PoseStamped, convertPoseStamped2Numpy
from gaussian_splatting_py.vision_node import VisionClass
from gaussian_splatting.srv import QueryPose, QueryPoseResponse, QueryPoseRequest
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest
from gaussian_splatting_py.frames_sam2 import run_video_prediction
from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.base import extract_object_pc

from kinova_control_py.status import *
from kinova_control_py.gripper_model import Gripper
from kinova_control_py.pose_util import calcAngDiff, convert_pose_to_pos_quat
from kinova_control_py.planner import CuroboPlanner
from kinova_control_py.tsdf_integration import O3DTsdfIntegration
from kinova_control_py.curobo_controller import CuroboController

OBJECT_CENTER = np.array([0.5, 0., 0.25])
TRAJ_STRUCT = namedtuple("Trajectory", ["position", "velocity"])
        
class CuroboROSWrapper(CuroboController):
    def __init__(self, robot_type:str, planner:str):
        super().__init__(robot_type, planner)
        rospy.init_node("planner")
        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        
        self.joint_control_pub = rospy.Publisher('/joint', JointState, queue_size=10)
        joint_state_msg = rospy.wait_for_message('/joint_state', JointState)
        self.current_state = np.array(joint_state_msg.position)[:7]
        self.home_pose = self.planner.calcFK(self.current_state)
        self.grasp_model = rospy.get_param("~grasp_model", "contact_graspnet")
        self.logger_fn = rospy.loginfo

        self.current_joint_sub = rospy.Subscriber('/joint_state', JointState, self.joint_callback)
        self.get_joint_state = lambda: self.current_state
        self.contral_rate = rospy.Rate(30)
        
        # wait for vision node service
        rospy.loginfo("Waiting for Vision Node Services...")
        rospy.wait_for_service("/add_view")
        rospy.wait_for_service("/next_best_view")
        rospy.wait_for_service("/save_model")

        self.add_view_client = rospy.ServiceProxy("/add_view", Trigger)
        self.nbv_client = rospy.ServiceProxy("/next_best_view", NBV)
        self.save_model_client = rospy.ServiceProxy("/save_model", Trigger)
        self.get_gs_data_dir_client = rospy.ServiceProxy("/get_gs_data_dir", Trigger)
        # self.add_touch_client = rospy.ServiceProxy("/add_touch", Trigger)
        self.query_pose_client = rospy.ServiceProxy("/grasp_pose", QueryPose)
        
        self.close_gripper = lambda: self.send_req_helper(self.close_gripper_clinet, TriggerRequest())
        self.open_gripper = lambda: self.send_req_helper(self.open_gripper_clinet, TriggerRequest())
        self.save_model = lambda: self.send_req_helper(self.save_model_client, TriggerRequest())
        self.add_view = lambda: self.send_req_helper(self.add_view_client, TriggerRequest())
        self.get_joint_velocity = lambda: np.array(rospy.wait_for_message("/joint_state", JointState).velocity)
        self.get_gs_data_dir = lambda : self.send_req_helper(self.get_gs_data_dir_client, TriggerRequest()).message
        
        self.open_gripper_clinet = rospy.ServiceProxy("/open_gripper", Trigger)
        self.close_gripper_clinet = rospy.ServiceProxy("/close_gripper", Trigger)
        
        rospy.loginfo("Vision Node Services are available")
        
    def joint_callback(self, msg:JointState):
        self.current_state = np.array(msg.position)[:7]    
    
    def traj_pub_command(self, position, velocity):
        joint_state_msg = JointState()
        joint_state_msg.position = list(position)
        joint_state_msg.velocity = list(velocity)
        self.joint_control_pub.publish(joint_state_msg)
        self.contral_rate.sleep()
           
    def send_req_helper(self, client, req):
        """ Send request helper with ROS service"""
        while True:
            res = client(req)

            if res.success:
                break
            else:
                # error analysis
                error_code = int(res.message.split(":")[0].split(" ")[2])

                if error_code in [2, 3]:
                    rospy.logwarn(res.message)
                    exit()
                elif error_code == 1:
                    # wait for training loop to be finised
                    rospy.loginfo(res.message)
                    rospy.sleep(1)
                else:
                    print(res.message)

        return res
    
    def predict_grasp(self, gaussian_splatting_data_dir):
        req = QueryPoseRequest()
        req.method = self.grasp_model
        req.datadir = gaussian_splatting_data_dir

        res = self.send_req_helper(self.query_pose_client, req)
        if len(res.poses) == 0:
            self.logger_fn("Fail to Predict Grasp Pose")
            return [], []
        
        grasp_poses = np.stack([convertPoseStamped2Numpy(pose) for pose in res.poses], axis=0)
        scores = np.array(res.scores)
        return grasp_poses, scores
    
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
        color_frame = "camera_color_frame" # if rs_image is None else "realsense_color_frame"
        ee_frame = "ee"
        # color_frame = "tool_frame"
        try:
            transform = self.tfBuffer.lookup_transform(ee_frame, color_frame, rospy.Time())
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            exit(1)
            
        # get the camera pose
        camera_pose = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        camera_quat = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])
        cam2hand = np.eye(4)
        cam2hand[:3, :3] = Quaternion(camera_quat).rotation_matrix
        cam2hand[:3, 3] = camera_pose
        
        candidate_cam_poses = []
        candidate_ee_poses = []
        
        trials = 0
        current_state = self.current_state
        while trials < num_poses * 10 and len(candidate_cam_poses) < num_poses:
            # sample a pose and use KDL to do IK
            pose = self.planner.sampleInSphere(center, 0.35, 0.5)
            
            hand_pose = np.eye(4)
            hand_pose[:3, 3] = pose[:3]
            hand_pose[:3, :3] = SciR.from_quat(pose[3:]).as_matrix()
            success, _ = self.planner.plan(hand_pose, current_state)
            
            if success:
                rospy.loginfo("Sample one candidate pose")
                cam_pose = hand_pose @ cam2hand
                pose_msg:PoseStamped = convertNumpy2PoseStamped(cam_pose)
                candidate_cam_poses.append(pose_msg)
                candidate_ee_poses.append(hand_pose)
            else:
                rospy.logwarn("Fail to plan trajectory, continue sampling ...")

        return candidate_ee_poses, candidate_cam_poses
    
    def vision_phase_active(self, gs_data_dir):
        dataset = KinovaDataset(gs_data_dir)
        object_pc = extract_object_pc(dataset)[1]
        object_center = np.mean(object_pc, axis=0)
        
        # update the observations
        self.planner.fuse_observation(gs_data_dir)
        
        trial = 0
        images_taken = 0
        while trial < 8 and images_taken < 4:
            req = NBVRequest()
            candidate_ee_poses, req.poses = self.generate_poses(20, object_center)
            res = self.send_req_helper(self.nbv_client, req)
            scores = np.array(res.scores)
            
            if np.max(scores) == np.min(scores):
                rospy.logwarn("No Valid Next Best View")
                trial += 1
                continue
            
            best_index = np.argmax(scores)
            
            best_pose = candidate_ee_poses[best_index]
            success, traj = self.planner.plan(best_pose, self.current_state)
            self.planner.go(traj, command_cb=self.traj_pub_command)
            
            req = TriggerRequest()
            self.send_req_helper(self.add_view_client, req)
            # propogate SAM2
            run_video_prediction(gs_data_dir)
            trial += 1
            images_taken += 1
            
        # Save the model
        rospy.loginfo("Save Model ...")
        req = TriggerRequest()
        self.send_req_helper(self.save_model_client, req)
    
        return True

    def run(self):
        # Phase 1 - 4 views sampled uniformly
        self.vision_phase_azimuth()
        gs_data = self.get_gs_data_dir()
        
        run_video_prediction(gs_data)
        torch.cuda.empty_cache()
        
        # Phase 2 - 4 views sampled actively
        self.vision_phase_active(gs_data)
        
        # Phase 3 - Grasp
        run_video_prediction(gs_data)
        torch.cuda.empty_cache()
        self.grasp_phase(gs_data)

if __name__ == "__main__":
    planner = CuroboROSWrapper("franka", "curobo")
    planner.run()
