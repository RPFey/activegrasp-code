# This is the script that uses Foundation Pose to grasp objects
# It could be served as a template for other grasp methods

from gaussian_splatting_py.foundation_pose_interface import FoundationPoseInterface
from kinova_control_py.gripper_model import Gripper

class FoundationPoseObjectGrasp:
    """ 
        This is the class for the grasp method using Foundation Pose 
        The idea is to record the grasping poses before experiment in the object frame
        During experiment, the model will predict the object pose and find the look up table for the candidate grasps. 
    """
    def __init__(self, mesh_file, recorded_poses,
                    score_model_path, refine_model_path):
        """  

            Args:
                mesh_file: the mesh file of the object (CAD Model)
                recorded_poses: the recorded grasping poses of the object
        """
        self.mesh_file = mesh_file
        self.recorded_poses = recorded_poses
        self.foundation_pose_interface = FoundationPoseInterface(mesh_file, score_model_path, refine_model_path)
        self.sample_coords = Gripper.load_candidate(recorded_poses)

    def predict_grasp_pose(self, **obs):
        """ 
            Predict the grasp pose of the object
        
            obs: the observation of the object
                dict: 
                    has keys 'gaussian_splatting_data_dir'
        """

        # do detection in the gaussian splatting dir
        gaussian_splatting_data_dir = obs['gaussian_splatting_data_dir']
        data_pack = self.foundation_pose_interface.infer_from_data_dir(gaussian_splatting_data_dir)

        # select the first 
        object_pose = None
        for result in data_pack:
            # select z upward
            if result["object_pose"][2, 2] > 0 :
                object_pose = result["object_pose"]
                break
        
        if object_pose is None:
            print("Fail to get the object pose")
            return None
        
        candidate_id = obs.get("grasp_id", 0)
        candidate = self.sample_coords[candidate_id]
        transform = object_pose @ candidate

        return transform
        
    def step(self, **observation):
        """ Interface for the grasp method """
        pass
