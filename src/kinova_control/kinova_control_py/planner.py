import numpy as np
from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere
from scipy.spatial.transform import Rotation as sciR

# Third Party
import torch
from abc import ABC, abstractmethod
from typing import Callable
import math
import tqdm
import threading

from gaussian_splatting_py.datasets import KinovaDataset

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.camera import CameraObservation
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig, CudaRobotModelState
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

def fabonacci_sphere(samples=1000):
    """
    Generate the points on a Fabonacci Sphere
    """
    rnd = 1.
    offset = 2. / samples
    increment = np.pi * (3. - np.sqrt(5.))

    points = []
    i = np.random.randint(0, samples)
    y = ((i * offset) - 1) + (offset / 2)
    r = np.sqrt(1 - pow(y, 2))

    phi = ((i + rnd) % samples) * increment

    x = np.cos(phi) * r
    z = np.sin(phi) * r

    return np.array([x, y, z]) 

def fabonacci_sphere_angle(samples=1000):
    """
    Return the angle from the Fabonacci Sphere
    """
    xyz = fabonacci_sphere(samples)
    
    theta = np.arctan2(xyz[1], xyz[0])
    if theta < 0:
        theta += 2 * np.pi

    phi = np.arccos(xyz[2])

    return theta, phi

class Planner(ABC):
    def __init__(self, robot_type:str):
        """
            Initialize the planner 
        """
        self.robot_type = robot_type
        self.rng = np.random.default_rng(12345)

    def setup_scene(self):
        """ 
            Set up the scene for the planner
        """
        # in x, y, z, qx, qy, qz, qw
        box_name = "desktop"
        # add box to the scene. In the future, resize to object size in GS
        BOX_DIMS = (4, 4, 1)
        box_pose = np.array([0, 0, -(BOX_DIMS[2]) / 2 - 0.01, 0, 0, 0, 1])
        self.add_object(box_name, box_pose, 'box', size=BOX_DIMS)
        
        # wall_name = "wall"
        # BOX_DIMS = (1, 1, 1)
        # wall_pose = np.array([-(BOX_DIMS[0]) / 2 - 0.45, 0, 0, 0, 0, 0, 1])
        # self.add_object(wall_name, wall_pose, 'box', size=BOX_DIMS)

        # ceil_name = "ceil"
        # BOX_DIMS = (4, 4, 0.2)
        # ceil_pose = np.array([0, 0, 10. + BOX_DIMS[2] / 2, 0, 0, 0, 1])
        # self.add_object(ceil_name, ceil_pose, 'box', size=BOX_DIMS)
    
    def toggle_collision_for_grasp(self, flag = False):
        """ Toggle the collision for the grasp """
        pass

    @abstractmethod
    def add_object(self, object_name:str, object_pose:np.ndarray, type:str, **args):
        """  """
        pass
    
    def sampleOnSphere(self, radius, center_pos,
                     min_phi = 0., max_phi = np.pi / 4,
                     min_theta = np.pi / 2, max_theta = 3 * np.pi / 2):
      """ Sample a point on the sphere with given center and radius 
      
      Args:
        radius (float): The radius of the sphere
        center_pos (np.ndarray): The center position of the sphere
        angle phi: 0 - up, pi / 2 - horizontal
        angle theta: 0 - front, pi - back
      """
      # sample a point on the sphere
      theta = min_theta + self.rng.random() * (max_theta - min_theta)
      phi = min_phi + self.rng.random() * (max_phi - min_phi)
        
      return self.poseOnSphere(theta, phi, radius, center_pos)
    
    def fibonacci_sample(self, radius, center_pos,
                        min_phi = 0., max_phi = np.pi / 2,
                        min_theta = np.pi / 2, max_theta = 3 * np.pi / 2):
        """ Sample a point on the sphere using Finoacci Sample
        
        Args:
        radius (float): The radius of the sphere
        center_pos (np.ndarray): The center position of the sphere
        angle phi: 0 - up, pi / 2 - horizontal
        angle theta: 0 - front, pi - back
        """
        assert min_phi >= 0 and max_phi <= np.pi , "The phi angle should be within [0, pi/2]"
        assert min_phi < max_phi - np.pi / 180, "The min_phi should be smaller than max_phi"

        assert min_theta >= 0 and max_theta <= 2 * np.pi, "The theta angle should be within [0, 2 * pi]"
        assert min_theta < max_theta  - np.pi / 180, "The min_theta should be smaller than max_theta"

        # sample a point on the sphere
        while True:
            theta, phi = fabonacci_sphere_angle()
            if theta >= min_theta and theta <= max_theta and phi >= min_phi and phi <= max_phi:
                break

        return self.poseOnSphere(theta, phi, radius, center_pos)
        
    def sampleInSphere(self, center_pos,
                        min_radius = 0.01, max_radius = 0.1,
                        min_phi = 0, max_phi = np.pi / 3,
                        #  min_phi = np.pi / 4., max_phi = np.pi / 2,
                        min_theta = np.pi / 2, max_theta = 3 * np.pi / 2):
        """ Sample the pose within the sphere """

        radius = min_radius + self.rng.random() * (max_radius - min_radius)
        return self.sampleOnSphere(radius, center_pos, min_phi, max_phi, min_theta, max_theta)

    def poseOnSphere(self, theta, phi, radius, center_pos, scalar_first=False):
        """ generate a pose on the sphere
        
        Args:
        radius (float): The radius of the sphere
        center_pos (np.ndarray): The center position of the sphere
        angle phi: 0 - up, pi / 2 - horizontal
        angle theta: 0 - front, pi - back
        """
        offset = np.array([radius * np.sin(phi) * np.cos(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(phi)])

        # TODO Move Fabonacci Sphere to generate angles.
        # offset = fabonacci_sphere()
        # offset[:2] = abs(offset[:2]) # z axis is up
        # offset *= radius
        pos = center_pos + offset

        # compute the orientation
        vec_z = -1 * offset / np.linalg.norm(offset)
        
        # y axis is horizontal
        if np.isnan(vec_z[0] / vec_z[1]):
            vec_x = np.array([0, 1., 0.])
        else:
            vec_x = np.array([1, -vec_z[0] / vec_z[1], 0])
            vec_x = vec_x / np.linalg.norm(vec_x)

        # x_axis
        vec_y = np.cross(vec_z, vec_x)

        if vec_y[2] < 0:
            vec_x = -vec_x
            vec_y = -vec_y

        R_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)

        # get the quaternion
        quat = sciR.from_matrix(R_matrix).as_quat(scalar_first=scalar_first)    
        pose = np.concatenate((pos, quat))

        return pose

class CuroboPlanner(Planner):
    def __init__(self, robot_type:str):
        """
            Initialize the planner 
        """
        super(CuroboPlanner, self).__init__(robot_type)
        # self.world_cuboid = []
        # self.world_mesh = []
        self.robot_type = robot_type
        self.world_cfg = WorldConfig.from_dict(
            {
                "blox": {
                    "world": {
                        "pose": [0, 0, 0, 1, 0, 0, 0],
                        "integrator_type": "occupancy",
                        "voxel_size": 0.02,
                    }
                }
            }
        )
        self.setup_scene()

        if self.robot_type == 'franka':
            self.config_file_path =  "franka.yml"
        elif self.robot_type == 'kinova':
            self.config_file_path = "kinova_gen3.yml"

        # convenience function to store tensor type and device
        self.tensor_args = TensorDeviceType()

        # this example loads urdf from a configuration file, you can also load from path directly
        # load a urdf, the base frame and the end-effector frame:
        config_file = load_yaml(join_path(get_robot_path(), self.config_file_path))

        urdf_file = config_file["robot_cfg"]["kinematics"][
            "urdf_path"
        ]  # Send global path starting with "/"
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        # Generate robot configuration from  urdf path, base frame, end effector frame
        self.robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)
        
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.config_file_path,
            self.world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.BLOX,
            use_cuda_graph=True,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.03,
            collision_activation_distance=0.01,
            acceleration_scale=1.0,
            self_collision_check=True,
            maximum_trajectory_dt=0.25,
            finetune_dt_scale=1.05,
            fixed_iters_trajopt=True,
            finetune_trajopt_iters=300,
            minimize_jerk=True,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        self.motion_gen.warmup(warmup_js_trajopt=False)
        
    def clean_map(self):
        """ Clean the map """
        self.motion_gen.world_collision.clear_cache()

    def add_object(self, object_name:str, object_pose:np.ndarray, type:str, **args):
        """  """
        if type == 'box':
            size = args['size']
            cube = Cuboid(
                name=object_name,
                pose=[object_pose[0], object_pose[1], object_pose[2], object_pose[6], object_pose[3], object_pose[4], object_pose[5]],
                dims=[size[0], size[1], size[2]],
                color=[0.8, 0.0, 0.0, 1.0],
            )
            self.world_cfg.add_obstacle(cube)
        
        if type == "mesh":
            scale = args.get("scale", [1., 1., 1.])
            mesh = Mesh(
                name=object_name,
                pose=[object_pose[0], object_pose[1], object_pose[2], object_pose[6], object_pose[3], object_pose[4], object_pose[5]],
                file_path=args['path'],
                scale=scale,
            )
            self.world_cfg.add_obstacle(mesh)

    def plan(self, goal_pose:np.ndarray, start_state:np.ndarray, constrained = False):
        """
            Plan the trajectory
            
            Args:
                goal_pose: np.ndarray of shape (4, 4) or (7,)
                start_state: np.ndarray of shape (7,)
                constrained: bool, whether to use constrained optimization, the ee will be constrained to the goal pose in z - direction
        """
        goal_cupose = CuroboPlanner.convert_to_pose(goal_pose, self.tensor_args)
        
        joint_state = JointState(
            position=torch.tensor(start_state[None, :], device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            velocity=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            acceleration=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
        )
        
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.25,
        )
        
        if constrained:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
            )
            plan_config.pose_cost_metric = pose_cost_metric
        
        # Use MotionGen to plan trajectory
        # import pdb; pdb.set_trace()
        result = self.motion_gen.plan_single(joint_state, goal_cupose, plan_config)    
        
        if not result.success:
            print(result.status)
            return False, None
            
        traj = result.get_interpolated_plan() # result.interpolation_dt has the dt between timesteps
        return result.success, traj
    
    def plan_batch(self, goal_pose:np.ndarray, start_state:np.ndarray, batch_size:int = 4):
        """
            Plan from the current start state to the goal pose
        """
        num_goals = goal_pose.shape[0]
        batch_num = int(math.ceil(num_goals / batch_size))
        
        plan_config = MotionGenPlanConfig(
            enable_graph=True,
            enable_graph_attempt=4,
            max_attempts=2,
            enable_finetune_trajopt=True
        )
        
        for i in range(batch_num):
            low = i * batch_size
            high = min((i + 1) * batch_size, num_goals)
            
            current_goal_pose = goal_pose[low: high, ...]
            goal_cupose = CuroboPlanner.convert_to_pose(current_goal_pose, self.tensor_args)
            
            start_state = torch.tensor(start_state, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            joint_state = JointState(
                position=start_state[None, None, :].repeat(high - low, 1, 1),
                velocity=torch.zeros((high - low, 1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
                acceleration=torch.zeros((high - low, 1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            )
            
            result = self.motion_gen.plan_batch(joint_state, goal_cupose, plan_config)
            
            if torch.any(result.success):
                success_idx = torch.min(torch.where(result.success)[0])
                plan = result.interpolated_plan[success_idx].trim_trajectory(0, result.path_buffer_last_tstep[success_idx])
                return success_idx.item() + low, plan
        
        return -1, None
    
    def plan_grasp(self, goal_pose:np.ndarray, start_state:np.ndarray):
        """
            Plan the trajectory
            
            Args:
                goal_pose: np.ndarray of shape (4, 4) or (7,)
                start_state: np.ndarray of shape (7,)
                constrained: bool, whether to use constrained optimization, the ee will be constrained to the goal pose in z - direction
        """ 
        goal_cupose = CuroboPlanner.convert_to_pose(goal_pose, self.tensor_args, batchify=True)
        
        joint_state = JointState(
            position=torch.tensor(start_state[None, :], device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            velocity=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            acceleration=torch.zeros((1, 7), device=self.tensor_args.device, dtype=self.tensor_args.dtype),
        )
        
        plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=2,
            enable_finetune_trajopt=True,
            time_dilation_factor=0.25,
        )
        
        # Use MotionGen to plan trajectory
        # import pdb; pdb.set_trace()
        result = self.motion_gen.plan_grasp(joint_state, goal_cupose, 
                                                plan_config)    
        
        if not result.success:
            print(result.status)
            return False, None
        
        return result.success, result
    
    def fuse_observation(self, datadir, start_id = 0):
        dataset = KinovaDataset(datadir)
        world_model = self.motion_gen.world_collision
        
        num_views = len(dataset)
        for i in range(start_id, num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K']
            c2w = data_pack['camtoworld'].numpy()

            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            
            position = torch.from_numpy(c2w[:3, 3])
            orientation = sciR.from_matrix(c2w[:3, :3]).as_quat(scalar_first=True)
            
            camera_pose = Pose(
                position=self.tensor_args.to_device(position),
                quaternion=self.tensor_args.to_device(orientation),
            )

            # should be c2w
            data_camera = CameraObservation(  # rgb_image = data["rgba_nvblox"],
                depth_image=depth, intrinsics=cam_K, pose=camera_pose
            )
            
            data_camera = data_camera.to(device=self.tensor_args.device)
            world_model.add_camera_frame(data_camera, "world")
            world_model.process_camera_frames("world", False)
            torch.cuda.synchronize()
            world_model.update_blox_hashes()
                
    def go(self, traj, command_cb: Callable):
        positions = traj.position.cpu().numpy()
        velocities = traj.velocity.cpu().numpy()
        
        for p, v in tqdm.tqdm(zip(positions, velocities)):
            command_cb(p, v)
            
        return True
    
    def calcFK(self, qs:np.ndarray):
        """
            Calculate the forward kinematics
        """
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        
        assert qs.shape[1] == self.kin_model.dof, "Input joint angles should be of shape (n, {})".format(self.kin_model.dof)

        qs_t = torch.tensor(qs, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        ee_pose = self.kin_model.get_state(qs_t)
        ee_pose = CuroboPlanner.convert_to_numpy(ee_pose)
        return ee_pose[0]
    
    def toggle_collision_for_grasp(self, flag = False):
        if self.robot_type == 'franka':
            self.motion_gen.toggle_link_collision(["panda_leftfinger", "panda_rightfinger"], flag)
    
    @staticmethod
    def convert_to_pose(poses:np.ndarray, tensor_args:TensorDeviceType = None, batchify = False):
        if tensor_args is None:
            tensor_args = TensorDeviceType()

        ee_position = []
        ee_quaternion = [] # w, x, y, z
        
        if poses.ndim >= 2 and poses.shape[-1] == 4 and poses.shape[-2] == 4:
            poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                ee_position.append(pose[:3, 3])
                ee_quaternion.append(sciR.from_matrix(pose[:3, :3]).as_quat(scalar_first=True))
        
        elif poses.ndim >= 2 and poses.shape[-1] == 7:
            poses = poses.reshape(-1, 7)
            for pose in poses:
                ee_position.append(pose[:3])
                ee_quaternion.append(pose[3:])
        
        elif poses.ndim == 1 and poses.shape[0] == 7:
            ee_position.append(poses[:3])
            ee_quaternion.append(poses[3:])

        ee_position = torch.tensor(np.array(ee_position), device=tensor_args.device, dtype=tensor_args.dtype)
        ee_quaternion = torch.tensor(np.array(ee_quaternion), device=tensor_args.device, dtype=tensor_args.dtype)
        
        if batchify:
            return Pose(ee_position.view(1, -1, 3), ee_quaternion.view(1, -1, 4), batch=ee_position.shape[0])
        else:
            if len(ee_position) == 1:
                return Pose(ee_position, ee_quaternion, batch=ee_position.shape[0])
            else:
                return Pose(ee_position.view(-1, 1, 3), ee_quaternion.view(-1, 1, 4), batch=ee_position.shape[0])

    @staticmethod
    def convert_to_numpy(ee_pose:CudaRobotModelState):
        ee_position = ee_pose.ee_position.cpu().numpy()
        ee_quaternion = ee_pose.ee_quaternion.cpu().numpy()

        ts = []
        for p, q in zip(ee_position, ee_quaternion):
            r = sciR.from_quat(q, scalar_first=True)
            q = r.as_matrix()

            t = np.eye(4)
            t[:3, :3] = q
            t[:3, 3] = p
            ts.append(t)

        return np.stack(ts)

    
    def calcIK(self, poses:np.ndarray):
        """
        Args:
            poses: np.ndarray of shape (n, 4, 4) or (n, 7) 
                quat should be in w, x, y, z format
        """ 
        
        goal = CuroboPlanner.convert_to_pose(poses, self.tensor_args)
        result = self.motion_gen.ik_solver.solve_batch(goal)
        
        return result.success, result.solution
    
    def process_joints(self, joint_angles:np.ndarray):
        """
            Process the joint angles
        """
        joint_limits = self.kin_model.get_joint_limits().position.cpu().numpy()
        joint_angles = np.where(joint_angles > joint_limits[1], joint_angles - 2 * np.pi, joint_angles)
        joint_angles = np.where(joint_angles < joint_limits[0], joint_angles + 2 * np.pi, joint_angles)
        return joint_angles
    
# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20    

def example_move_to_start_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    constrained_joint_angles = Base_pb2.ConstrainedJointAngles()

    actuator_count = base.GetActuatorCount().count
    angles = [0.0] * actuator_count

    # Actuator 4 at 90 degrees
    for joint_id in range(len(angles)):
        joint_angle = constrained_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Reaching joint angles...")
    base.PlayJointTrajectory(constrained_joint_angles)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Joint angles reached")
    else:
        print("Timeout on action notification wait")
    return finished

def example_forward_kinematics(base):
    # Current arm's joint angles (in home position)
    try:
        print("Getting Angles for every joint...")
        input_joint_angles = base.GetMeasuredJointAngles()
    except KServerException as ex:
        print("Unable to get joint angles")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False

    print("Joint ID : Joint Angle")
    for joint_angle in input_joint_angles.joint_angles:
        print(joint_angle.joint_identifier, " : ", joint_angle.value)
    print()
    
    # Computing Foward Kinematics (Angle -> cartesian convert) from arm's current joint angles
    try:
        print("Computing Foward Kinematics using joint angles...")
        pose = base.ComputeForwardKinematics(input_joint_angles)
    except KServerException as ex:
        print("Unable to compute forward kinematics")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        return False

    print("Pose calculated : ")
    print("Coordinate (x, y, z)  : ({}, {}, {})".format(pose.x, pose.y, pose.z))
    print("Theta (theta_x, theta_y, theta_z)  : ({}, {}, {})".format(pose.theta_x, pose.theta_y, pose.theta_z))
    print()
    return True

# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def get_joint_angles(base):
    joint_angles = [p.value for p in base.GetMeasuredJointAngles().joint_angles] # .joint_angles
    
    # wrap joint angles to [-pi, pi]
    joint_angles = np.array(joint_angles) / 180 * np.pi
    
    while np.any(joint_angles > np.pi):
        joint_angles = np.where(joint_angles > np.pi, joint_angles - 2 * np.pi, joint_angles)
    
    while np.any(joint_angles < -np.pi):
        joint_angles = np.where(joint_angles < -np.pi, joint_angles + 2 * np.pi, joint_angles)    
    
    return joint_angles

if __name__ == '__main__':
    import argparse
    import kinova.utilities as utilities
    args = argparse.ArgumentParser()
    # args.add_argument('--datadir', type=str, default='/home/yangchen/curobo/data/kinova_data')
    args.add_argument('--robot', type=str, default="franka")
    opt = utilities.parseConnectionArguments(args)
    # opt = args.parse_args()
    
    # test kinova planner
    planner = CuroboPlanner(opt.robot)
        
    if opt.robot == 'kinova':
        from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
        from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
        from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
        from kortex_api.autogen.messages import Session_pb2, Base_pb2, Common_pb2

        with utilities.DeviceConnection.createTcpConnection(opt) as router:

            # Create required services
            base = BaseClient(router)

            import pdb; pdb.set_trace()
            # Example core
            success = True
            home = planner.process_joints(
                    np.array([360, 15, 180, 230, 0, 55, 90]) / 180 * np.pi        
            )
            home_pose = planner.calcFK(home)
            # success &= example_move_to_start_position(base)
            
            import pdb; pdb.set_trace()
            example_forward_kinematics(base)
            joint_angles = get_joint_angles(base)
            print(planner.calcFK(joint_angles))
            ret, traj = planner.plan(home_pose, joint_angles)
            
            joint_speeds = Base_pb2.JointSpeeds()
            actuator_count = base.GetActuatorCount().count
            
            # The 7DOF robot will spin in the same direction for 10 seconds
            position, velocity = traj.position.cpu().numpy(), traj.velocity
            for p, v in zip(traj.position, traj.velocity):
                
                speeds = [SPEED, 0, -SPEED, 0, SPEED, 0, -SPEED]
                i = 0
                for speed in speeds:
                    joint_speed = joint_speeds.joint_speeds.add()
                    joint_speed.joint_identifier = i 
                    joint_speed.value = speed
                    joint_speed.duration = 0
                    i = i + 1
                
                print ("Sending the joint speeds for 10 seconds...")
                base.SendJointSpeedsCommand(joint_speeds)
                time.sleep(0.01)
            
            # success &= example_send_joint_speeds(base)
            base.Stop()
            
        