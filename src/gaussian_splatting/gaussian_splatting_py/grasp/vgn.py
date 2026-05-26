import time
import numpy as np
import collections
import argparse
from pathlib import Path
import open3d as o3d
import cv2
from numba import jit

from gaussian_splatting_py.datasets import KinovaDataset
from gaussian_splatting_py.grasp.vgn_src.grasp import *
from gaussian_splatting_py.grasp.vgn_src.network import *
from gaussian_splatting_py.grasp.vgn_src.perception import *
from gaussian_splatting_py.grasp.vgn_src.transform import Transform, Rotation
from gaussian_splatting_py.grasp.base import GraspEstimator, extract_point_clouds, extract_object_pc
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image

State = collections.namedtuple("State", ["tsdf", "pc"])
Intrinsic = collections.namedtuple("Intrinsic", ["width", "height", "fx", "fy", "cx", "cy"])

@jit(nopython=True)
def get_voxel_at(voxel_size, p, voxel_dim):
    index = (p / voxel_size).astype(np.int64)
    
    if (index >= 0).all() and (index < voxel_dim).all():
        return index.astype(np.int64)
    else:
        return np.array([-1, -1, -1], np.int64)

@jit(nopython=True)
def raycast(
    voxel_size,
    tsdf_grid,
    ori,
    pos,
    fx,
    fy,
    cx,
    cy,
    u_min,
    u_max,
    v_min,
    v_max,
    t_min,
    t_max,
    t_step,
    origin
):
    voxel_indices = []
    voxel_dim = tsdf_grid.shape[0]
    for u in range(u_min, u_max):
        for v in range(v_min, v_max):
            direction = np.asarray([(u - cx) / fx, (v - cy) / fy, 1.0])
            direction = ori @ (direction / np.linalg.norm(direction))
            t, tsdf_prev = t_min, -1.0
            while t < t_max:
                p = pos + t * direction - origin
                t += t_step
                i, j, k = get_voxel_at(voxel_size, p, voxel_dim)
                if i >= 0:
                    tsdf = tsdf_grid[i, j, k]
                    if tsdf * tsdf_prev < 0 and tsdf_prev > -1:  # crossed a surface
                        break
                    voxel_indices.append([i, j, k])
                    tsdf_prev = tsdf
    return voxel_indices

class VGN(object):
    def __init__(self, model_path, rviz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)
        self.rviz = rviz

    def __call__(self, state):
        tsdf_vol = state.tsdf.get_grid()
        voxel_size = state.tsdf.voxel_size

        tic = time.time()
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        grasps, scores = select(qual_vol.copy(), rot_vol, width_vol, threshold=0.9)
        toc = time.time() - tic

        grasps, scores = np.asarray(grasps), np.asarray(scores)

        if len(grasps) > 0:
            p = np.random.permutation(len(grasps))
            grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps[p]]
            scores = scores[p]

        # if self.rviz:
        #     vis.draw_quality(qual_vol, state.tsdf.voxel_size, threshold=0.01)

        return grasps, scores, toc


def predict(tsdf_vol, net, device):
    assert tsdf_vol.shape == (1, 40, 40, 40)

    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        qual_vol, rot_vol, width_vol = net(tsdf_vol)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    rot_vol = rot_vol.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, rot_vol, width_vol


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > 0.5
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < 0.5)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    # threshold on grasp quality
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score

class VGNGrasp(GraspEstimator):
    """ 
        
    """
    def __init__(self, args):
        """  
            args contains keys:
                finger_depth : float
                model_path : path to vgn model
        """
        self.finger_depth = args.finger_depth
        self.voxel_size = 6.0 * self.finger_depth 
        
        self.vgn = VGN(Path(args.model_path))
        # self.compile()

    def compile(self):
        # Trigger the JIT compilation
        raycast(
            1.0,
            np.zeros((40, 40, 40)),
            np.eye(3),
            np.zeros(3),
            1.0,
            1.0,
            1.0,
            1.0,
            0,
            100,
            0,
            100,
            0.0,
            1.0,
            0.1,
            np.zeros((3,))
        )
    
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
        
        # compute the first frame
        # set the ground to -0.04 to include the table
        object_pcs = extract_object_pc(dataset)[1]
        min_coord, max_coord = np.min(object_pcs, axis=0), np.max(object_pcs, axis=0)
        min_coord[2] = -0.04
        bound = np.max(max_coord - min_coord)
        
        low_res_tsdf = TSDFVolume(self.voxel_size, 40, origin = min_coord)
        high_res_tsdf = TSDFVolume(self.voxel_size, 120, origin = min_coord)
        
        num_views = len(dataset)
        for i in range(num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K, pc_full, pc_colors = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], None, None
            c2w = data_pack['camtoworld'].numpy()

            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            segmap, rgb, depth, cam_K = 1 - segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy()
            
            camera_intrinsic = CameraIntrinsic(W, H, cam_K[0, 0], cam_K[1, 1], 
                                                    cam_K[0, 2], cam_K[1, 2])
            
            w2c = np.linalg.inv(c2w)
            low_res_tsdf.integrate(depth, camera_intrinsic, w2c)
            high_res_tsdf.integrate(depth, camera_intrinsic, w2c)
            
        tsdf = low_res_tsdf
        # this pc is under world coord frame
        pc = high_res_tsdf.get_cloud()
        
        # predict grasps
        state = State(tsdf, pc)
        grasps, scores, planning_time = self.vgn(state)
        
        if len(grasps) == 0:
            print("No grasps found")
            return [], {}
    
        # The grasp poses are under the TSDF coord frame, should add the origin of TSDF
        # to transform them to world coord frame
        t = np.array([[0., -1., 0., 0.], [1., 0., 0., 0.], [0, 0., 1., 0.], [0., 0., 0., 1.]])
        grasp_poses = np.stack([p.pose.as_matrix() @ t for p in grasps], axis=0)
        grasp_poses[:, :3, 3] += min_coord[None, :]
        
        pts_cnt = np.array([self.count_pts_in_gripper(object_pcs, g, gripper_depth=self.finger_depth, gripeer_width=0.1) for g in grasp_poses])
        pts_thresh = 1e4
        print("[WARN] filter grasp poses based on pts inside gripper, threshold set to {}".format(pts_thresh))
        mask = pts_cnt > pts_thresh
        
        if np.sum(mask) == 0:
            print("No grasp found")
            return [], {}
        
        grasp_poses = grasp_poses[mask].reshape(-1, 4, 4)
        scores = scores[mask]
        
        # change base to wrist, to align with results from contact grasp net and se3diff
        grasp_poses[:, :3, 3] -= grasp_poses[:, :3, 2] * self.finger_depth / 2
        
        score_sort_idx = np.argsort(scores)[::-1]  
        grasp_poses = grasp_poses[score_sort_idx]
        object_score = scores[score_sort_idx]
        
        viz = obs.get('viz', False)
        if viz:
            data_pack = dataset[0]
            segmap, rgb, depth, cam_K = \
                    data_pack["mask"], data_pack['image'], data_pack['depths'], data_pack['K']
            segmap, rgb, depth, cam_K = 1 - segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy()
            c2w = data_pack['camtoworld'].numpy()
            
            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                    skip_border_objects=True, 
                                                                    z_range=[0, 2.])   
            pc_full = pc_full @ c2w[:3, :3].T + c2w[:3, 3:4].T             
            visualize_grasps(pc, {1: grasp_poses}, {1: scores}, pc_colors=pc_colors)
            
        return grasp_poses, {"scores": object_score}
    
    def compute_IG(self, c2ws, dataset:KinovaDataset):
        """ 
        Function to compute Information Gain based on Breyer's paper
            https://arxiv.org/pdf/2207.10543
        
        Args:
            c2ws: [np.ndarray] the views of the object (N, 4, 4), c2ws
            dataset: [KinovaDataset] the dataset of the object
            
        Returns:
            scores: [np.ndarray] the scores of the views
        """
        # find pointcloud for object
        object_pcs = extract_object_pc(dataset)[1]
        
        obj_min_coord, obj_max_coord = np.min(object_pcs, axis=0), np.max(object_pcs, axis=0)
        # set the ground to -0.04 to include the table
        # some other expansion
        task_min_coord = obj_min_coord - np.array([0.1, 0.1, 0])
        task_min_coord[2] = -0.04
        task_max_coord = obj_max_coord + np.array([0.1, 0.1, 0])
        dim = 96 # int(np.max( (task_max_coord - task_min_coord) ) / self.voxel_size)
        
        # integrate TSDF
        tsdf = TSDFVolume(np.max(task_max_coord - task_min_coord), dim, origin = task_min_coord)
        num_views = len(dataset)
        for i in range(num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K, pc_full, pc_colors = \
                    data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], None, None
            c2w = data_pack['camtoworld'].numpy()

            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            segmap, rgb, depth, cam_K = 1 - segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy()
            
            camera_intrinsic = CameraIntrinsic(W, H, cam_K[0, 0], cam_K[1, 1], 
                                                    cam_K[0, 2], cam_K[1, 2])
            
            w2c = np.linalg.inv(c2w)
            tsdf.integrate(depth, camera_intrinsic, w2c)   
        
        data_pack = dataset[0]
        rgb, cam_K =  data_pack['image'], data_pack['K']
        cam_K = cam_K.numpy()
        H, W, C = rgb.shape
        intrinsic = CameraIntrinsic.from_dict(
            {"width": W, "height": H, "K": cam_K.reshape(-1)}
        )
        
        # compute IG for each view
        scores = []
        for c2w in c2ws:
            score = self.ig_fn(c2w, tsdf, intrinsic, 8, (obj_min_coord, obj_max_coord))
            scores.append(score)
        
        return scores
            
    def ig_fn(self, c2w, tsdf:TSDFVolume, intrinsic:CameraIntrinsic,   
                downsample = 8, bbox = None):
        """ 
            Compute IG for one single view 
            
            Args:
                c2w: [np.ndarray] the camera to world transformation
                tsdf: [TSDFVolume] the TSDF volume of the object
                intrinsic: [CameraIntrinsic] the intrinsic of the camera
                downsample: [int] the downsample factor
                bbox: [Tuple] the bounding box of the object (min_coord, max_coord)
        """
        tsdf_grid, voxel_size, origin = tsdf.get_grid(), tsdf.voxel_size, tsdf.origin.astype(np.float64)
        tsdf_grid = -1.0 + 2.0 * tsdf_grid[0]  # Open3D maps tsdf to [0,1]
        
        # get the bbox of the volume
        min_coord, size = tsdf.origin, tsdf.size
        corners_idx = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        corners = min_coord + corners_idx * size # (8, 3)
        
        fx, fy, cx, cy = intrinsic.fx / downsample, intrinsic.fy / downsample, \
                            intrinsic.cx / downsample, intrinsic.cy / downsample
        
        w2c = np.linalg.inv(c2w)
        cornres_c = corners @ w2c[:3, :3].T + w2c[:3, 3:4].T
        u = fx * cornres_c[:, 0] / cornres_c[:, 2] + cx
        u = np.clip(u, 0, intrinsic.width / downsample - 1)  
        v = fy * cornres_c[:, 1] / cornres_c[:, 2] + cy
        v = np.clip(v, 0, intrinsic.height / downsample - 1)
        
        u_min, u_max = int(np.floor(np.min(u))), int(np.ceil(np.max(u)))
        v_min, v_max = int(np.floor(np.min(v))), int(np.ceil(np.max(v)))
        print("Image x range: {} - {}, y range: {} - {}".format(u_min, u_max, v_min, v_max))
        
        t_min = 0.0  # self.min_z_dist
        t_max = cornres_c[:, 2].max()  # This bound might be a bit too short
        t_step = np.sqrt(3) * voxel_size  # Could be replaced with line rasterization

        voxel_indices = raycast(
            voxel_size,
            tsdf_grid,
            w2c[:3, :3],
            w2c[:3, 3],
            fx,
            fy,
            cx,
            cy,
            u_min,
            u_max,
            v_min,
            v_max,
            t_min,
            t_max,
            t_step,
            origin
        )

        if len(voxel_indices) == 0:
            return 0
        
        # Count rear side voxels within the bounding box
        voxel_indices = np.array(voxel_indices, dtype=np.int64)
        indices = np.unique(voxel_indices, axis=0)
        
        if bbox is not None:
            min_coord, max_coord = bbox
            min_coord[2] = -0.04
            bbox_min = (min_coord - origin) / voxel_size
            bbox_max = (max_coord - origin) / voxel_size
            mask = np.array([((i > bbox_min) & (i < bbox_max)).all() for i in indices])
            i, j, k = indices[mask].T
        else:
            i, j, k = indices.T
            
        tsdfs = tsdf_grid[i, j, k]
        ig = np.logical_and(tsdfs > -1.0, tsdfs < 0.0).sum()
        
        return ig
    
    def count_pts_in_gripper(self, pc, grasp, gripper_depth=0.1, gripeer_width=0.2):
        """ Count the number of points inside the gripper """
        world2gripper = np.linalg.inv(grasp)
        pc_in_hand = pc @ world2gripper[:3, :3].T + world2gripper[:3, 3].T
        
        mask_z = np.bitwise_and(pc_in_hand[:, 2] <= gripper_depth, pc_in_hand[:, 2] > 0)
        mask_y = np.bitwise_and(pc_in_hand[:, 0] <= gripeer_width / 2, pc_in_hand[:, 2] >= -gripeer_width / 2)
        mask_x = np.bitwise_and(pc_in_hand[:, 1] <= gripper_depth / 2, pc_in_hand[:, 1] >= -gripeer_width / 2)
        mask = mask_x & mask_y & mask_z
        
        pts_cnt = np.sum(mask)
        return pts_cnt

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--model_path", type=str, default="/root/Backup/src/gaussian_splatting/weights/vgn_conv.pth")
    arg.add_argument("--np_path", type=str, default="/root/data/chip")
    arg.add_argument("--finger_depth", type=float, default=0.05)
    opt = arg.parse_args()
    
    v = VGNGrasp(opt)
    v.predict_grasp_pose(data_path=opt.np_path, viz=True)