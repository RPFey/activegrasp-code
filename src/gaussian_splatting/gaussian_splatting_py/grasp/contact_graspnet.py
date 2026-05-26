# This is the script that uses Contact Grasp Net to grasp objects
# It could be served as a template for other grasp methods

import glob
import os
import argparse
import torch
import open3d as o3d
import numpy as np
import collections
import cv2

from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.base import GraspEstimator, extract_point_clouds, extract_pcd_batch

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator as ContactGraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.config_utils import load_config

import multiprocessing as mp
import time
import viser
from einops import rearrange, repeat, reduce

def viz_pc(xyz: np.array, port:int=12333, color=None):
    def viz_fn():
        colors = np.zeros_like(xyz)
        if color is not None:
            colors[:] = color
        else:   
            colors[:, 0] = 1
        server = viser.ViserServer(port=port)
        base_frame = server.scene.add_frame(
            "/frames",
            # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=True,
        )
        server.scene.add_point_cloud(
                    "/frames/point_cloud",
                    points=xyz,
                    point_size=0.001,
                    colors=colors,
                )
        while True:
            time.sleep(0.01)
    p = mp.Process(target=viz_fn)
    p.start()
    return p

CONTACTGRASP_CONFIG = collections.namedtuple(
  "CONTACTGRASP_CONFIG", ["ckpt_dir", "z_range", "local_regions", "filter_grasps", "skip_border_objects", "forward_passes"], 
)

class ContactGraspNet(GraspEstimator):
    """ 
        
    """
    def __init__(self, global_config, args):
        """  

            Args:
                mesh_file: the mesh file of the object (CAD Model)
                recorded_poses: the recorded grasping poses of the object
        """
        # Build the model
        self.grasp_estimator = ContactGraspEstimator(global_config)
        self.global_config = global_config
        self.args = args

        # Load the weights
        model_checkpoint_dir = os.path.join(self.args.ckpt_dir, 'checkpoints')
        checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=self.grasp_estimator.model)
        try:
            load_dict = checkpoint_io.load('model.pt')
        except FileExistsError:
            print('No model checkpoint found')
            load_dict = {}

    @torch.no_grad()
    def predict_grasp_pose(self, **obs):
        """ 
            Predict the grasp pose of the object
        
        Args:
            obs: the observation of the object
                dict: has keys 'data_path', 'viz'
                    if it has 'data_path', then it will read the dataset in KinovaDataset format
                    otherwise, it should contain the dataset key;
                        which is a list of dictionaries with keys 'mask', 'image', 'depths', 'K', 'camtoworld'
                        [
                            {"mask": ..., "image": ..., "depths": ..., "K": ..., "camtoworld": ...},
                            ...
                        ]
        Returns:
            poses: [List] the predicted grasp poses
            meta: dict 
                "scores" : the scores of the predicted grasp poses
                "contact_pts": the contact points of the predicted grasp poses
        """
        # take the first observation
        if 'data_path' in obs:
            dataset = KinovaDataset(obs['data_path'], touch_cfg=obs.get('touch_cfg', 'config/digit.yaml'))
        else:
            dataset = obs['dataset']

        # TODO robot camera flip
        # Contact GraspNet requires the camera frame to be
        # in a specific pose.
        add_transform = np.eye(4)
        add_transform[:, 0] = np.array([0, 1, 0, 0])
        add_transform[:, 1] = np.array([-1, 0, 0, 0])
        use_prot_comp = False
        pc_full, pc_colors, fuse_pc_segments, base_transform = \
                            extract_pcd_batch(dataset, add_transform, 
                                    use_prot_comp=use_prot_comp, include_scene_info = True)
        
        if use_prot_comp:
            self.grasp_estimator._contact_grasp_cfg['TEST']['filter_thres'] = 0.01 # because comp_pcs is not that dense

        # run prediction
        print('Generating Grasps...')
        del fuse_pc_segments[0] # remove the background
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(pc_full, 
                                                                                       pc_segments=fuse_pc_segments, 
                                                                                       local_regions=self.args.local_regions, 
                                                                                       filter_grasps=self.args.filter_grasps, 
                                                                                       forward_passes=self.args.forward_passes)
        torch.cuda.empty_cache()
        # breakpoint()
        viz = obs.get('viz', False)
        if viz:
            pc_full, pc_colors = pc_full[::8], pc_colors[::8]
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

        # since the id in mask is 1, we use '1' to query the info
        object_grasp_pose = pred_grasps_cam[1]
        object_score = scores[1]
        contact_pts = contact_pts[1]
        
        if len(object_grasp_pose) == 0:
            return [], {}

        poses = np.stack([base_transform @ g for g in object_grasp_pose], axis=0)
        contact_pts = np.stack([base_transform[:3, :3] @ c + base_transform[:3, 3] for c in contact_pts], axis=0)

        # sort the poses by the scores
        score_sort_idx = np.argsort(object_score)[::-1]  
        poses = poses[score_sort_idx]
        object_score = object_score[score_sort_idx]
        contact_pts = contact_pts[score_sort_idx]

        meta = {
            "scores": object_score,
            "contact_pts": contact_pts
        }
        
        return poses, meta
        
    def step(self, **observation):
        """ Interface for the grasp method """
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='/root/contact_graspnet_pytorch/checkpoints/contact_graspnet', help='Log dir')
    parser.add_argument('--np_path', default='/root/data/chip', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--z_range', default=[0.1, 1.], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=True, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=5,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    contact_network = ContactGraspNet(global_config, FLAGS)
    grasp_pose = contact_network.predict_grasp_pose(data_path=FLAGS.np_path, viz=True)

    