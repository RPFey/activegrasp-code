import numpy as np
import rospy
from scipy.spatial.transform import Rotation as sciR

from geometry_msgs.msg import PoseStamped, Pose

def convertNumpy2PoseStamped(pose:np.ndarray) -> PoseStamped:
    """ Convert Numpy to PoseStamped 
      If pose is 1D, it is a 7D pose, in the order of x y z qx qy qz qw
      
      If pose is 2D, it is a 4x4 pose
    """
    pose_msg = PoseStamped()
    # relative to world
    pose_msg.header.frame_id = "base_link"
    pose_msg.header.stamp = rospy.Time.now()

    # convert R7 to msg
    if pose.ndim == 1:
      pose_msg.pose.position.x = pose[0]
      pose_msg.pose.position.y = pose[1]
      pose_msg.pose.position.z = pose[2]
      pose_msg.pose.orientation.x = pose[3]
      pose_msg.pose.orientation.y = pose[4]
      pose_msg.pose.orientation.z = pose[5]
      pose_msg.pose.orientation.w = pose[6]
    
    # convert 4x4 to msg
    elif pose.ndim == 2:
      pose_msg.pose.position.x = pose[0, 3]
      pose_msg.pose.position.y = pose[1, 3]
      pose_msg.pose.position.z = pose[2, 3]

      Rot = sciR.from_matrix(pose[:3, :3])
      quat = Rot.as_quat()

      pose_msg.pose.orientation.x = quat[0]
      pose_msg.pose.orientation.y = quat[1]
      pose_msg.pose.orientation.z = quat[2]
      pose_msg.pose.orientation.w = quat[3]

    return pose_msg

def convertPoseStamped2Numpy(pose_msg:PoseStamped) -> np.ndarray:
    """ Convert PoseStamped to Numpy 4x4 matrix """
    pose = np.zeros((4, 4))
    pose[3, 3] = 1.
    pose[0, 3] = pose_msg.pose.position.x
    pose[1, 3] = pose_msg.pose.position.y
    pose[2, 3] = pose_msg.pose.position.z

    quat = np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w])
    Rot = sciR.from_quat(quat)
    pose[:3, :3] = Rot.as_matrix()
    
    return pose