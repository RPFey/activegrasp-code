# This script generates 6D grasp poses for a scene point cloud 
# and use viser to visualize the scene and the generated grasps.
# Usage:
# python scripts/sample/generate_scene_6d_grasp_poses.py
#
# Specify the scene as .npz in the viser gui.
# npz file should contain "scene_pts" (N, 3) and "target_index" (N,) where target_index == 1 for target points and 0 for scene points.
import viser
import torch
import argparse
import time
import os

import open3d as o3d
from pytorch3d.ops import sample_farthest_points
import scipy.spatial.transform
import numpy as np
import matplotlib.pyplot as plt

from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD, Grasp_PCSampler
from se3dif.utils import to_numpy, to_torch, load_experiment_specifications
from se3dif.visualization import grasp_visualization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

class ViserVisualizer:
    def __init__(self, port):
        self.server = viser.ViserServer(port=port, verbose=False)
        self.scale = 8.
        self._add_names = []
        self.num_grasps = 16
        
        self.model_spec_handle = self.server.gui.add_text(
            "Model Spec",
            "/path/to/spec"
        )
        self.model_spec_handle.on_update(self._on_model_spec_change)
        
        self.model_weight_handle = self.server.gui.add_text(
            "Model Weight",
            "/path/to/ckpt"
        )
        self.model_weight_handle.on_update(self._on_model_weight_change)
        
        self.text_handle = self.server.gui.add_text(
            "Scene File",
            "Scene"
        )
        self.text_handle.on_update(self._on_text_change)
        
        self.grasp_number_handle = self.server.gui.add_number(
            "Grasp Number",
            self.num_grasps
        )
        self.grasp_number_handle.on_update(self._on_grasp_number_change)
        
        self.model = None
        self.sampler = None
        self._pts = None
        self._label = None
        
    def _on_text_change(self, x):
        print(f"File Load: {x.target.value}")
        
        if not os.path.exists(x.target.value):
            print(f"File {x.target.value} does not exist")
            return

        try:
            data = np.load(x.target.value)
            scene_pts = data['scene_pts']
            target_index = data['target_index']
        except KeyError as e:
            print(f"File {x.target.value} does not contain the required keys")
            return
        except (FileNotFoundError, IsADirectoryError) as e:
            print(f"File {x.target.value} not found")
            return
        
        # clean the scene
        self._clean()
        
        # downsample & normalize
        target_pts = ViserVisualizer.fps_sample(scene_pts[target_index == 1], 512)
        scene_pts = ViserVisualizer.fps_sample(scene_pts[target_index == 0], 512)
        target_mean = np.mean(target_pts, axis=0)
        target_pts = (target_pts - target_mean) * self.scale
        scene_pts = (scene_pts - target_mean) * self.scale
        pts = np.concatenate([target_pts, scene_pts], axis=0)
        label = np.concatenate([np.ones(target_pts.shape[0]), np.zeros(scene_pts.shape[0])], axis=0)
    
        # Create the point cloud
        colors = np.zeros((pts.shape[0], 3))  # Random RGB colors in [0,1] range
        colors[label == 1, 0] = 1.0
        colors[label == 0, 2] = 1.0
        self._add_element("/scene_pcd", self.server.scene.add_point_cloud, pts / self.scale, colors, 0.001)
        # self.server.scene.add_point_cloud("scene_pcd", points=pts / self.scale, colors=colors, point_size=0.001)
        # self._add_names.append("scene_pcd")
        
        self._pts = to_torch(pts).to(device).view(1, -1, 3)
        self._label = to_torch(label).to(device).view(1, -1, 1)
        
        if self.model is not None and self.sampler is not None:
            self.model.set_latent(self._pts, self._label)
            grasps, energy = self.sampler.sample()
        
            grasps = to_numpy(grasps)  
            # negate energy so that low energy -> bright color
            energy = -1 * to_numpy(energy).reshape(-1)
            self.visualize_grasp(grasps, energy)
        
    def visualize_grasp(self, grasps, energy):
        """
        Visualize the grasps:
            grasps: (num_grasps, 4, 4) array of grasps
            energy: (num_grasps,) array of energy values
        """
        # if self.model.distribution == 'direct':
        #     energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-6)
        # else:
        #     energy = np.exp(energy)
        
        cmap = plt.get_cmap('viridis')
        # in scale of 0 - 1
        colors = cmap(energy)
        
        for i, (H, c) in enumerate(zip(grasps, colors)):
            gripper = grasp_visualization.create_gripper_marker()
            # H[:, -1] *= 1 / self.scale
            gripper.apply_transform(H)
            
            # paint color for mesh
            gripper.visual.vertex_colors = np.tile(c, (len(gripper.vertices), 1))
            self._add_element(f"/grasp/gripper_{i}", self.server.scene.add_mesh_trimesh, gripper)
            # self.server.scene.add_mesh_trimesh(f"gripper_{i}", gripper)
            
    def _on_model_spec_change(self, x):
        print(f"Model Spec: {x.target.value}")
        # clean grasps
        for i in range(self.num_grasps):
            self._remove_name_if_exists(f"/grasp/gripper_{i}")
            
        model_args = load_experiment_specifications(x.target.value)
        model_args['device'] = device
        self.model = load_model(model_args)
        self.model.to(device)
        self.sampler = Grasp_PCSampler(self.model, batch=self.num_grasps, T=70, T_fit=30, device=device)
        self.model_weight_handle.value = "/path/to/ckpt"
        print(" Model Constructed ! ")  
    
    def _on_model_weight_change(self, x):
        try:
            if not os.path.exists(x.target.value):
                print(f"File {x.target.value} does not exist")
                return
            
            model_states = torch.load(x.target.value, map_location=device)
        except (FileNotFoundError, IsADirectoryError) as e:
            print(f"File {x.target.value} not found")
            return
            
        # clean grasps
        for i in range(self.num_grasps):
            self._remove_name_if_exists(f"/grasp/gripper_{i}")
            
        print(f"Model Weight: {x.target.value}")
        self.model.load_state_dict(model_states["model_state"])
        
        if self._pts is not None and self._label is not None:
            self.model.set_latent(self._pts, self._label)
            grasps, energy = self.sampler.sample()
            grasps = to_numpy(grasps)  
            # negate energy so that low energy -> bright color
            energy = -1 * to_numpy(energy).reshape(-1)
            self.visualize_grasp(grasps, energy)
        
    @staticmethod 
    def fps_sample(points, num_pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsample = pcd.farthest_point_down_sample(num_pts)
        return np.asarray(downsample.points)

    def _remove_name_if_exists(self, name):
        if name in self._add_names:
            self.server.scene.remove_by_name(name)
            self._add_names.remove(name)
    
    def _add_element(self, name, func, *element, **args):
        self._remove_name_if_exists(name)
        func(name, *element, **args)
        self._add_names.append(name)
        
    def _clean(self):
        for name in self._add_names:
            self.server.scene.remove_by_name(name)
        self._add_names.clear()
        
    def _on_grasp_number_change(self, x):
        # clean grasps
        for i in range(self.num_grasps):
            self._remove_name_if_exists(f"/grasp/gripper_{i}")
            
        self.num_grasps = x.target.value
        self.sampler = Grasp_AnnealedLD(self.model, batch=self.num_grasps, T=30, T_fit=50, device=device)

        if self._pts is not None and self._label is not None:
            self.model.set_latent(self._pts, self._label)
            grasps, energy = self.sampler.sample()
            grasps = to_numpy(grasps)  
            # negate energy so that low energy -> bright color
            energy = -1 * to_numpy(energy).reshape(-1)
            self.visualize_grasp(grasps, energy)

def main(args):
    server = ViserVisualizer(args) 
    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    main(args)