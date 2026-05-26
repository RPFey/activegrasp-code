import time
import numpy as np
from pyquaternion import Quaternion
from math import pi
from scipy.spatial.transform import Rotation as sciR


# import rospy
# from geometry_msgs.msg import PoseStamped, Pose
# from std_srvs.srv import Empty
# import kdl_parser_py.urdf as kdl_parser
# import PyKDL

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

def convert_pose_to_pos_quat(pose, scalar_first=False):
    """ 
    Convert the pose to quaternion

    Arg:  
      pose np.ndarray (4x4)
    Return:
      pose (7,) x, y, z, qx, qy, qz, qw
    """

    pos = pose[:3, 3]
    quat = sciR.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate((pos, quat))

class PoseSolver(object):
  def __init__(self,
               base_name="base_link", ee_name="tool_frame",
               cache_size=100):
    robot_description = rospy.get_param(rospy.get_namespace() + "/robot_description")
    ret, kdl_tree = kdl_parser.treeFromString(robot_description)

    if not ret:
      rospy.logerr("Could not parse the URDF")
      return

    self.chain = kdl_tree.getChain(base_name, ee_name)
    self.fk = PyKDL.ChainFkSolverPos_recursive(self.chain)
    self.ik = PyKDL.ChainIkSolverPos_LMA(self.chain)
    self.JacSolver = PyKDL.ChainJntToJacSolver(self.chain)

    self.rng = np.random.default_rng(12345)

    # joints, pose, visit count
    # pre-defined some cache data
    self.cache_result = [
      [(-0.1128, -0.2298, -3.0260, -2.0978, 0.0404, -0.99, 1.53), (0.3088, -0.0099, 0.2494, 0.7079, 0.6921, 0.0975, 0.1013), 1],
      [(-5.9122e-5, 0.2602, 3.1399, -2.2700, 9.6080e-5, 0.9598, 1.5701), (0.6561, 0.0023, 0.4341, 0.4997, 0.5, 0.5, 0.5), 1],
      [(-0.9158, -1.3684, 2.9055, 1.8424, -0.3789, 1.35099, 1.36266), (0.061, -0.198, 0.040, 0.358, 0.639, 0.599, 0.323), 1],
      [(0.72758, 0.12090, 2.2582, -1.9493, 0.28768, -0.49335, 1.253075), (0.476, 0.035, 0.335, 0.667, 0.676, 0.231, 0.211), 1],
      [(-1.2635, 1.299707, 2.98492, -1.67081, 0.84283, -1.70457, -1.64795), (0.244, 0.209, 0.066, 0.720, 0.168, 0.221, 0.636), 1],
      [(-1.3559, 1.45078, 2.98523, -1.0148, 0.89388, -2.0485, -1.27655), (0.242, 0.367, 0.067, 0.721, 0.169, 0.220, 0.635), 1],
      [(-1.7206, 1.34562, 2.81540, -2.1716, 0.82419, -1.4242, -2.16406), (0.120, 0.131, 0.067, 0.672, 0.337, 0.359, 0.553), 1],

      # hard code view poses 
      # [(1.20210, 0.441493, -2.7450, -1.53978, -0.74564, -1.72046, -2.31065), (0.209, -0.346, 0.341, 0.390, 0.842, 0.327, 0.179), 1],
      # [(0.72814, -0.36849, 2.77811, -2.10217, -0.19593, -0.83717, 1.671975), (0.476, 0.232, 0.140, 0.798, 0.053, 0.040, 0.600), 1],
      # [(0.70222, -1.90703, 1.38093, -1.79663, -0.43980, -1.23098, 0.038853), (0.179, 0.245, 0.075, 0.644, 0.366, 0.266, 0.617), 1],
      # [(1.71299, -1.82577, 1.59172, -0.36620, -0.31521, -2.07172, -0.19647), (0.340, 0.418, 0.076, 0.694, 0.223, 0.141, 0.670), 1],
      # [(1.56034, -1.53714, 1.52106, -1.28326, -0.44244, -1.40506, 0.148811), (0.398, 0.287, 0.139, 0.822, 0.241, 0.093, 0.507), 1],
      # [(-0.65333, 0.22659, 2.40709, -1.41474, 0.669585, -2.05769, -0.61390), (0.227, 0.200, 0.474, 0.892, 0.293, 0.079, 0.336), 1],
      # [(-1.09604, -1.54194, -1.696842, -1.12810, 0.500093, -1.65155, -3.10903), (0.205, -0.466, 0.217, 0.316, 0.794, 0.453, 0.255), 1],
      # [(-1.9928, -1.60925, -0.4684, -0.15839, 2.787091, 2.099968, 0.405584), (0.419, -0.408, 0.079, 0.216, 0.833, 0.508, 0.034), 1]
    ]
    self.cache_size = cache_size

  def calcJac(self, joints) -> np.ndarray:
    """ Calculate the Jacobian matrix for the given joint positions 
    
    Args:
      joints (List[float]): The joint positions
    
    Return:
      np.ndarray: The Jacobian matrix (6, num_joints)
    """
    assert len(joints) == self.chain.getNrOfJoints(), "Joint Mismatch; the chain has {} joints, \
                                                        while the input has {} joins".format(self.chain.getNrOfJoints(), len(joints))
    
    q = PyKDL.JntArray(len(joints))
    for i in range(len(joints)):
      q[i] = joints[i]
    jac = PyKDL.Jacobian(len(joints))
    self.JacSolver.JntToJac(q, jac)

    jac_np = np.array([jac.getColumn(k) for k in range(jac.columns())])
    jac_np = jac_np.T # (6, num_joints)
    return jac_np
  
  @staticmethod
  def computePoseDistance(pose1, pose2):
    """ Compute the distance between two poses """
    assert len(pose1) == 7 and len(pose2) == 7, "The input poses should have 7 elements"
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    p1 = np.array(pose1[:3])
    p2 = np.array(pose2[:3])
    
    q1 = np.array(pose1[[6, 3, 4, 5]]) # change to qw, qx, qy, qz
    q2 = np.array(pose2[[6, 3, 4, 5]])

    # copmute quaternion distance
    quat1 = Quaternion(q1)
    quat2 = Quaternion(q2)

    rel_q = quat1.inverse * quat2
    theta = rel_q.radians

    return np.linalg.norm(p1 - p2) + theta

  def __storeCache(self, joint, pose):
    """ Store the joint and pose in the cache """
    
    # check the distance with poses inside cache

    # if the cache is full, remove the least visit element
    if len(self.cache_result) >= self.cache_size:
      visits = [stats[2] for stats in self.cache_result]
      visits = np.array(visits)
      min_idx = np.argmin(visits)

      # remove
      self.cache_result.pop(min_idx)

    self.cache_result.append([joint, pose, 1])

  def calcFK(self, joints) -> np.ndarray:
    """ Calculate the Forward Kinematics for the given joint positions
    
    Args:
      joints (List[float]): The joint positions
    
    Return:
      np.ndarray: The pose (x y z qx qy qz qw)
    """
    assert len(joints) == self.chain.getNrOfJoints(), "Joint Mismatch; the chain has {} joints, \
                                                        while the input has {} joins".format(self.chain.getNrOfJoints(), len(joints))
    q = PyKDL.JntArray(len(joints))
    for i in range(len(joints)):
      q[i] = joints[i]
    frame = PyKDL.Frame()
    self.fk.JntToCart(q, frame)
    
    pos = np.array([frame.p[k] for k in range(3)])
    quat = np.asarray(frame.M.GetQuaternion())
    pose = np.concatenate((pos, quat))
    return pose
  
  def calcIK(self, pose, init_joints=None) -> np.ndarray:
    """ Calculate the Inverse Kinematics for the given pose
    
    Args:
      pose (List[float]): The pose (x y z qx qy qz qw)

    Return:
      np.ndarray: The joint positions
    """
    if len(pose) == 7:
      pass
    elif len(pose.shape) == 2:
      assert pose.shape[0] == 4 and pose.shape[1] == 4, "Pose Mismatch; the input pose should have 4x4 elements"
      pose = convert_pose_to_pos_quat(pose)
    else:
      raise Exception("The input pose should have 7 elements or 4x4 elements")

    frame = PyKDL.Frame()
    frame.p = PyKDL.Vector(pose[0], pose[1], pose[2])
    frame.M = PyKDL.Rotation.Quaternion(pose[3], pose[4], pose[5], pose[6])
    q = PyKDL.JntArray(self.chain.getNrOfJoints())
    init_qs = []
  
    if init_joints is not None:
      init_q = PyKDL.JntArray(self.chain.getNrOfJoints())
      for i in range(self.chain.getNrOfJoints()):
        init_q[i] = init_joints[i]
      init_qs.append(init_q)
    
    elif len(self.cache_result) > 0:
      rospy.loginfo(" Use Cache result {} for solving ".format(len(self.cache_result)))
      distances = []
      for joint, ee, visit in self.cache_result:
        distance = RandomPoseGenerator.computePoseDistance(ee, pose)
        distances.append(distance)

      distances = np.array(distances)
      # put into the init qs
      sorted_args = np.argsort(distances)
      for ind in sorted_args:
        init_q = PyKDL.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
          init_q[i] = self.cache_result[ind][0][i]
          # increase visit
          self.cache_result[ind][2] += 1
        init_qs.append(init_q)

    # iterate through all cache results
    for init_q in init_qs:
      rospy.loginfo(" Try IK solver with {} results".format(len(init_qs)))
      ret = self.ik.CartToJnt(init_q, frame, q)
      
      if ret >= 0:
        joints = np.array([q[i] for i in range(q.rows())])
        
        cache_joints = np.array([stats[0] for stats in self.cache_result])
        joint_distance = np.linalg.norm(cache_joints - joints, axis=1)
        if np.min(joint_distance) > 0.1:
          self.__storeCache(joints, pose)
        
        return joints
      
    rospy.logerr("IK failed")
    return None

class RandomPoseGenerator(PoseSolver):
    def __init__(self, 
                 base_name="base_link", 
                 ee_name="tool_frame",
                cache_size=100 ):
        super().__init__(base_name, ee_name, cache_size)

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
      # return self.sampleOnSphere(radius, center_pos, min_phi, max_phi, min_theta, max_theta)
      return self.fibonacci_sample(radius, center_pos, min_phi, max_phi, min_theta, max_theta)

    def poseOnSphere(self, theta, phi, radius, center_pos):
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
      quat = sciR.from_matrix(R_matrix).as_quat()    
      pose = np.concatenate((pos, quat))

      return pose

def calcAngDiff(R_des, R_curr):
    """
    Calculate the angular difference between two rotation matrices.
    """
    omega = np.zeros((3, ))
    
    delta_R = R_curr.T @ R_des 
    S = (delta_R - delta_R.T) / 2
    
    omega = np.array([S[2, 1], S[0, 2], S[1, 0]])
    omega = R_curr @ omega
    return omega

if __name__ == '__main__':
  solver = PoseSolver()
  pose = np.array([0.1, 0.1, 0.1, 0, 0, 0, 1])
  joints = solver.calcIK(pose)
  