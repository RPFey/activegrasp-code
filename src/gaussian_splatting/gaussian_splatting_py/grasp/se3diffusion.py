# This is the script that uses Contact Grasp Net to grasp objects
# It could be served as a template for other grasp methods
from sksparse.cholmod import Factor
import glob
import os
import time
import argparse
import torch
import open3d as o3d
import numpy as np
import collections
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import logging

from einops import rearrange, repeat, reduce

from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.base import GraspEstimator, extract_point_clouds, extract_pcd_batch, depth2pc
from gaussian_splatting_py.tools.grasp_viser import ViserVisualizer
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.data import farthest_points, \
    distance_by_translation_point, preprocess_pc_for_inference, regularize_pc_point_count_index

from se3dif.models.loader import load_model
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD, Grasp_PCSampler
from se3dif.utils import SO3_R3, get_pretrained_models_src, load_experiment_specifications
from pytorch3d.ops import sample_farthest_points

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

class SE3Diffusion(GraspEstimator):
    """ 
        
    """
    def __init__(self, args, model_params, num_grasps = 512):
        """  
            Args:
                args contains keys:
                    z_range, local_regions, filter_grasps, skip_border_objects
                model_params: the model parameters to load, se3diff, se3diff_ap
                num_grasps: the number of grasps to sample
        """
        self.args = args

        # Build the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cp_90_score = -1. # 0.55
        self.num_grasps = num_grasps

        # Build & Construct Model
        model_src_path = get_pretrained_models_src()
        logging.info(os.path.join(model_src_path, model_params))
        model_args = load_experiment_specifications(os.path.join(model_src_path, model_params))
        model_args["device"] = self.device
        self.model = load_model(model_args)
        model_states = torch.load(os.path.join(model_src_path, model_params, 'model.pth'), map_location=self.device)
        if 'model_state' in model_states:
            self.model.load_state_dict(model_states['model_state'])
        else:
            self.model.load_state_dict(model_states)
            
        self.bg_expansion_factor = 1.5

    # @torch.no_grad()
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
            dataset = KinovaDataset(obs['data_path'])
        else:
            dataset = obs['dataset']
            
        scale = 8.
        pc_full, pc_colors, fuse_pc_segments, base_transform = extract_pcd_batch(dataset, include_scene_info=True)
        object_mean = np.mean(fuse_pc_segments[1], axis=0)

        if obs.get('pcs_from_splats', False):
            logging.info(f"transforming splats to pc")
            with torch.no_grad():
                splats = obs["splats"]
                splat_means = splats["means"].clone()
                object_pc, target_index = self.splats_to_pcs(splat_means, splats, base_transform)
        else:
            target_num_pts = 512 if self.model.num_scene_points > 0 else 1024
            scene_num_pts = 512
            logging.info(f"preprocess object_pc: {fuse_pc_segments[1].shape}")
        
            # voxel downsample
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(fuse_pc_segments[1])
            pcd = pcd.voxel_down_sample(0.002)
            fuse_pc_segments[1] = np.asarray(pcd.points)

            clustering = DBSCAN(eps=0.05, min_samples=50).fit(fuse_pc_segments[1])
            labels, cnts = np.unique(clustering.labels_, return_counts=True)
            target_label = labels[np.argmax(cnts)]
            fuse_pc_segments[1] = fuse_pc_segments[1][clustering.labels_ == target_label]
            object_pc_centralized, object_mean = preprocess_pc_for_inference(fuse_pc_segments[1], target_num_pts, None, use_farthest_point=True, return_mean=True)   
            
            # concatenate the object point cloud with the surrounding point cloud
            extent = np.max( np.max(object_pc_centralized, axis=0) - np.min(object_pc_centralized, axis=0) )
            distance = np.linalg.norm(fuse_pc_segments[0] - object_mean, axis=1)
            surr_pts = fuse_pc_segments[0][distance < self.bg_expansion_factor * extent]
            logging.info(f"preprocess surr_pts: {surr_pts.shape}")
            # surround_pc_centralized = preprocess_pc_for_inference(surr_pts, scene_num_pts, object_mean, use_farthest_point=True)
            # Use PyTorch3D to perform farthest point sampling on surr_pts
            surr_pts_tensor = torch.from_numpy(surr_pts).float().to(self.device).unsqueeze(0)  # Convert to tensor and add batch dimension
            _, sampled_indices = sample_farthest_points(surr_pts_tensor, K=scene_num_pts, random_start_point=True)
            surr_pts_sampled = surr_pts_tensor[0, sampled_indices[0]].cpu().numpy()  # Extract sampled points
            surround_pc_centralized = surr_pts_sampled - object_mean[None, :]  # Center the sampled points around the object mean

            complete_pc = np.concatenate([surround_pc_centralized, object_pc_centralized], axis=0)
            
            # append labels
            target_index = np.zeros((complete_pc.shape[0],), dtype=np.float32)
            target_index[-target_num_pts:] = 1
            
            if self.model.num_scene_points > 0:
                object_pc_normalize = complete_pc   
                target_index = torch.from_numpy(target_index).float().to(self.device)
                target_index = target_index.view(1, -1, 1)
            else:
                object_pc_normalize = object_pc_centralized
                target_index = None
                
            object_pc_normalize = object_pc_normalize # @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            object_pc_normalize = object_pc_normalize * scale
            object_pc = torch.from_numpy(object_pc_normalize).float().to(self.device)
            object_pc = object_pc.unsqueeze(0)
        
        logging.info(f"set_latent object_pc: {object_pc.shape}")

        ########### 2. SET SAMPLING METHOD #############
        self.model.set_latent(object_pc, target_index)
        generator = Grasp_AnnealedLD(self.model, batch=self.num_grasps, T=70, T_fit=50, k_steps=1, device=self.device, deterministic=False)
        H, energy = generator.sample()
        
        with torch.no_grad():
            canon_energy = energy / torch.exp(self.model.temperature) 
            success = torch.exp( -1 * canon_energy ) 
            entropy = -1 * ( success * torch.log(success + 1e-6) + (1 - success) * torch.log(1 - success + 1e-6) )
            eta_val = entropy.mean().item()
            print(f"grasp entropy: {eta_val}")
            
        H = H.cpu().numpy()
        H[:, :3, 3] = H[:, :3, 3] / scale + object_mean[None, :]
        if self.model.distribution != 'direct':
            score = torch.exp(-1 * energy / torch.exp(self.model.temperature) ).view(-1).detach().cpu().numpy()
        else:
            score = -1 * energy.view(-1).cpu().numpy()            
        
        if obs.get('viz', False):
            server = ViserVisualizer(port=8008)
            
            pc_full = complete_pc # pc_full[::8]
            pc_colors = np.zeros((complete_pc.shape[0], 3), dtype=np.float32)
            pc_colors[scene_num_pts:, 0] = 1
            pc_colors[:scene_num_pts, 2] = 1
            
            server._add_element("/scene_pcd", 
                              server.server.scene.add_point_cloud, pc_full, pc_colors, 0.005)
            H[:, :3, 3] = H[:, :3, 3] - object_mean[None, :]
            
            idx_sort = np.argsort(score)[::-1]
            H = H[idx_sort[:8]]
            score = score[idx_sort[:8]]
            score = (score - score.min()) / (score.max() - score.min())
            server.visualize_grasp( H, score )
            try:
                while True:
                    time.sleep(10.0)
            except KeyboardInterrupt:
                pass
            
        poses = np.stack([base_transform @ g for g in H], axis=0)

        # sort the poses by the scores
        score_sort_idx = np.argsort(score)[::-1]  
        poses = poses[score_sort_idx]
        object_score = score[score_sort_idx]

        meta = {
            "scores": object_score
        }
        
        return poses, meta
    
    def backpropagate(self, **obs):
        """ 
            Interface for the backpropagate method, the gradient is computed w.r.t the depth map
        """
        if 'data_path' in obs:
            dataset = KinovaDataset(obs['data_path'])
        else:
            dataset = obs['dataset']
        
        assert "splats" in obs, "The dataset should contain the key 'gs_pcs' which is the point cloud of the object"
        splats = obs["splats"]

        add_transform = np.eye(4)
        add_transform[0, 0] = -1
        add_transform[1, 1] = -1
        logging.info("extracting pcd batch")
        pc_full, pc_colors, fuse_pc_segments, base_transform = extract_pcd_batch(dataset, add_transform)
        
        # clustering to filter outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fuse_pc_segments[1])
        pcd = pcd.voxel_down_sample(0.002)
        fuse_pc_segments[1] = np.asarray(pcd.points)

        clustering = DBSCAN(eps=0.05, min_samples=50).fit(fuse_pc_segments[1])
        labels, cnts = np.unique(clustering.labels_, return_counts=True)
        target_label = labels[np.argmax(cnts)]
        fuse_pc_segments[1] = fuse_pc_segments[1][clustering.labels_ == target_label]

        if False:
            object_mean = np.mean(fuse_pc_segments[1], axis=0)
            logging.info("preprocess object_pc")
            object_pc_centralized = preprocess_pc_for_inference(fuse_pc_segments[1], 2048, object_mean, use_farthest_point=True)
            object_pc_normalize = object_pc_centralized * scale
            
            # object_pc_normalize = (fuse_pc_segments[1] - object_mean[None, :]) / scale
            object_pc = torch.from_numpy(object_pc_normalize).float().to(self.device)
            object_pc = object_pc.unsqueeze(0)
            logging.info("object_pc shape: {}".format(object_pc.shape))

        # get the grasp
        batch = 512
        logging.info("transforming spalts to pc")

        with torch.enable_grad():
            # compute the gradient w.r.t the depth map
            # take the first camera pose as the base transform
            splat_means = splats["means"].detach().clone()
            splat_means.requires_grad_(True)
            splat_means.retain_grad()
            splat_pc_normalize, target_index = self.splats_to_pcs(splat_means, splats, base_transform) 

            logging.info("infering latent")
            self.model.set_latent(splat_pc_normalize, target_index=target_index)
            for param in self.model.parameters():
                param.requires_grad_(True)

            generator = Grasp_AnnealedLD(self.model, batch=batch, T=100, T_fit=50, k_steps=1, device=self.device)
            grasp_poses, _ = generator.sample()
            logging.info(f"grasp_poses shape: {grasp_poses.shape}")
            
            ## 1.Set input variable to Theseus ##
            t_in = 1e-3 * torch.ones_like(grasp_poses[:, 0, 0])
            
            # TODO the step time might differ ?
            nll = self.model(grasp_poses, t_in)
            prob = torch.exp(-1 * nll).view(-1)
            energy = nll / torch.exp(self.model.temperature) 
            success = torch.exp( -1 * energy ) 
            entropy = -1 * ( success * torch.log(success + 1e-6) + (1 - success) * torch.log(1 - success + 1e-6) )
            
            # Sample - Diff
            # eta = torch.sum( entropy * prob ) / torch.sum(prob)
            # eta_val = eta.item()
            # eta.backward()
            
            # Diff - Exp
            # E[\nabla h] - E[h \nabla D] - E[h] E[\nabla D]
            surrogate_eta = entropy.mean() - (entropy.detach() * energy).mean() - entropy.detach().mean() * ( energy ).mean()
            eta_val = entropy.detach().mean().item()
            surrogate_eta.backward()
            
            # energy[0].backward()
            # fishers = torch.cat([grad_means3D, 
            #                     # rearrange(grad_sh, "n s c -> n (s c)"),
            #                     grad_opacities, grad_scales, grad_rotations], dim=1)

            J_grasp = torch.cat([
                splat_means.grad.detach(),
                # rearrange(torch.zeros_like(splats['opacities']), "n -> n 1"),
                # torch.zeros_like(splats['scales']),
                # torch.zeros_like(splats['quats'])
            ], dim=1)

            logging.info("J_grasp shape: {}".format(J_grasp.shape))

            
            # TODO: add viz to J^T J w.r.t grasp on the point cloud
            # Viz -- Viz 

            if False:
                with torch.no_grad():
                    cmap = plt.get_cmap('jet')

                    pc_grad = torch.norm(pc_grad, dim=-1)
                    pc_colors = cmap(pc_grad[0].cpu().numpy())[:, :3]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(object_pc_normalize[0].cpu().numpy())
                    pcd.colors = o3d.utility.Vector3dVector(pc_colors)
                    o3d.visualization.draw_geometries([pcd])
                sns.displot(J_grasp[:, 0].cpu().numpy(), kde=True)
                plt.savefig("J_grasp_dist.png")
            
            return eta_val, J_grasp

    def splats_to_pcs(self, splat_means, splats, base_transform, scale = 8.):
        
        with torch.no_grad():
            splat_sem = torch.sigmoid(rearrange(splats["semantics"], "n 1 -> n"))
            splat_means_np = splat_means.cpu().numpy()
            splat_opacities_np = torch.sigmoid(splats["opacities"]).cpu().numpy()

            # sample twice with farthest point sampling
            num_points = 512
            splat_idxs = torch.arange(splat_means.shape[0])

            fgd_idx_mask = np.logical_and(splat_sem.cpu().numpy() > 0., splat_opacities_np > 0.2)
            _, rel_fgd_pc_indices = regularize_pc_point_count_index(splat_means_np[fgd_idx_mask], num_points, use_farthest_point=True)
            fgd_pc_indices = splat_idxs[fgd_idx_mask][rel_fgd_pc_indices]

            dist_obj_center = np.linalg.norm(splat_means_np[fgd_idx_mask] - splat_means_np[fgd_idx_mask].mean(axis=0), axis=1)
            object_scale = np.quantile(dist_obj_center, 0.95)
            bgd_idx_mask = np.logical_and(splat_sem.cpu().numpy() <= 1e-3, splat_opacities_np > 0.2)
            
            # select the points that are within 2 * object_scale radius
            distances = np.linalg.norm(splat_means_np[bgd_idx_mask] - splat_means_np[fgd_idx_mask].mean(axis=0), axis=1)
            close_bgd_mask = distances < self.bg_expansion_factor * object_scale
            _, rel_bgd_pc_indices = regularize_pc_point_count_index(splat_means_np[bgd_idx_mask][close_bgd_mask], num_points, use_farthest_point=True)
            bgd_pc_indices = splat_idxs[bgd_idx_mask][close_bgd_mask][rel_bgd_pc_indices]

            # whole index after selection
            pc_indices = np.concatenate([bgd_pc_indices, fgd_pc_indices], axis=0)

            assert self.model.num_scene_points > 0, "current codebase only takes num_scane points > 0 but it's easy to change to accept num_scene_points = 0"
            target_index = torch.zeros((1, pc_indices.shape[0], 1), dtype=torch.float32).to(splat_means)
            target_index[:, bgd_pc_indices.shape[0]:, :] = 1

            # np.savetxt('fg_sampled_pts.txt', splat_means_np[fgd_pc_indices])
            # np.savetxt('bg_sampled_pts.txt', splat_means_np[bgd_pc_indices])

        #TODO: sample object pcs to 512 points and scene points within 2*object_scale radius to 512 both with FPS
        # transform from gs world frame to grasp world frame
        splat_to_grasp_transform = torch.from_numpy(np.linalg.inv(base_transform)).to(splat_means)
        splat_pc = splat_means @ splat_to_grasp_transform[:3, :3].T + splat_to_grasp_transform[:3, 3:4].T
        # splat_pc_viz = splat_means.detach().cpu().numpy() @ splat_to_grasp_transform[:3, :3].T  + splat_to_grasp_transform[:3, 3:4].T
        
        splat_pc = torch.index_select(splat_pc, 0, torch.from_numpy(pc_indices).long().cuda())
        splat_pc = splat_pc - splat_pc.mean(dim=0, keepdim=True)
        splat_pc_normalize = rearrange(splat_pc * scale, "n d -> 1 n d")
        
        return splat_pc_normalize, target_index

    def step(self, **observation):
        """ Interface for the grasp method """
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--np_path', default='/root/data/chip', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--z_range', default=[.1, .4], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=True, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=True,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    # parser.add_argument('--forward_passes', type=int, default=5,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    # parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    grasp_network = SE3Diffusion(FLAGS, "multiobject_scene_graspdif_dual", 128)
    grasp_pose = grasp_network.predict_grasp_pose(data_path=FLAGS.np_path, viz=True)

    # grasp_network.backpropagate(data_path=FLAGS.np_path)