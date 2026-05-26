
from abc import abstractmethod
import cv2
import os
import numpy as np
from tqdm import tqdm
import math
import torch
import open3d as o3d    

def depth2pc(depth, K, rgb=None, downsample = 1):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    if isinstance(depth, np.ndarray):
        mask = np.where(depth > 0)
        x,y = mask[1], mask[0]
        normalized_x = (x.astype(np.float32) - K[0,2])
        normalized_y = (y.astype(np.float32) - K[1,2])
    else:
        mask = torch.where(depth > 0)
        x,y = mask[1], mask[0]
        normalized_x = (x.float() - K[0,2])
        normalized_y = (y.float() - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
    
    if isinstance(depth, np.ndarray):
        pc = np.stack((world_x, world_y, world_z), axis=1)
    else:
        pc = torch.stack((world_x, world_y, world_z), dim=1)
    
    pc = pc[::downsample]
    if rgb is not None:
        rgb = rgb[::downsample]
        
    return (pc, rgb)

def mask_culling(pc, masks, w2cs, Ks, pc_color = None, valid_ratio = 0.5):
    """ Cull the point cloud based on the masks """
    visit_count = np.zeros(pc.shape[0], dtype=np.uint8)
    
    for mask, w2c, K in tqdm(zip(masks, w2cs, Ks), desc='Mask Culling'):
        
        # project the point cloud to the image plane
        pc_cam = pc @ w2c[:3, :3].T + w2c[:3, 3:4].T
        pc_u = pc_cam[:, 0] * K[0, 0] / pc_cam[:, 2] + K[0, 2]
        pc_v = pc_cam[:, 1] * K[1, 1] / pc_cam[:, 2] + K[1, 2]
        
        # check if the point is in the mask
        in_bound_mask = (pc_u >= 0) & (pc_u < mask.shape[1]) & (pc_v >= 0) & (pc_v < mask.shape[0])
        
        pc_u = np.clip(pc_u, 0, mask.shape[1] - 1).astype(np.int32)
        pc_v = np.clip(pc_v, 0, mask.shape[0] - 1).astype(np.int32)
        in_mask = mask[pc_v, pc_u] > 0
        
        # point cloud mask 
        visit_count[in_bound_mask & in_mask] += 1
    
    valid_cnt = int(math.ceil(len(masks) * valid_ratio))
    valid_mask = visit_count >= valid_cnt
    
    pc = pc[valid_mask]
    if pc_color is not None:
        pc_color = pc_color[valid_mask]
            
    if pc_color is not None:
        return pc, pc_color
    
    return pc

def extract_point_clouds(depth, K, segmap=None, rgb=None, 
                            z_range=[0.2,1.8], segmap_id=0, 
                            skip_border_objects=False, margin_px=5, downsample=1):
        """
        Converts depth map + intrinsics to point cloud. 
        If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

        Arguments:
            depth {np.ndarray} -- HxW depth map in m
            K {np.ndarray} -- 3x3 camera Matrix

        Keyword Arguments:
            segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
            rgb {np.ndarray} -- HxW rgb image (default: {None})
            z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
            segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
            skip_border_objects {bool} -- Skip segments that are at the border of 
                                                the depth map to avoid artificial edges (default: {False})
            margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

        Returns:
            [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
        """

        if K is None:
            raise ValueError('K is required either as argument --K or from the input numpy file')
            
        # Convert to pc 
        pc_full, pc_colors = depth2pc(depth, K, rgb, downsample)

        # Threshold distance
        if pc_colors is not None:
            pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
        pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
        
        # Extract instance point clouds from segmap and depth map
        pc_segments = {}
        if segmap is not None:
            pc_segments = {}
            obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>=0])
            for i in obj_instances:
                if skip_border_objects and not i==segmap_id:
                    obj_i_y, obj_i_x = np.where(segmap==i)
                    
                    # Check if object is in image bounds
                    if np.any(obj_i_x < margin_px) or \
                            np.any(obj_i_x > segmap.shape[1]-margin_px) or \
                                np.any(obj_i_y < margin_px) or \
                                    np.any(obj_i_y > segmap.shape[0]-margin_px):
                        print('object {} not entirely in image bounds, skipping'.format(i))
                        continue
                
                inst_mask = segmap==i
                pc_segment, _ = depth2pc(depth*inst_mask, K)
                pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

        return pc_full, pc_segments, pc_colors
    
def extract_object_pc(dataset, skip_border_objects = False, 
                            z_range = [0.2, 1.8], downsample = 1):    
    """ Extract the point cloud for the object from the dataset using Mask """
    fuse_pc_segments = {}

    num_views = len(dataset)
    for i in range(num_views):
        data_pack = dataset[i]
        segmap, rgb, depth, cam_K, c2w = \
                data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K'], data_pack['camtoworld']

        H, W, C = rgb.shape
        depth = depth.reshape(H, W)
        if isinstance(segmap, torch.Tensor):
            segmap, rgb, depth, cam_K, c2w = \
                segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy(), c2w.numpy()

        # print('Converting depth to point cloud(s)...')
        _, pc_segments, _ = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb, 
                                                    skip_border_objects=skip_border_objects, z_range=z_range,
                                                    downsample=downsample)       
        
        # it seems that Contact GraspNet is not SE(3)-Equivariant ... 
        rel = c2w
        for k, v in pc_segments.items():
            pc_segments[k] = v @ rel[:3, :3].T + rel[:3, 3:4].T

        for k, v in pc_segments.items():
            if k not in fuse_pc_segments:
                fuse_pc_segments[k] = []
            fuse_pc_segments[k].append(v)

    # concatenate point cloud from different view points
    for k, v in fuse_pc_segments.items():
        fuse_pc_segments[k] = np.concatenate(v, axis=0)

    return fuse_pc_segments

def extract_pcd_batch(dataset, add_transform = np.eye(4), 
                            skip_border_objects = False, z_range = [0.2, 1.8], 
                            use_prot_comp = False, downsample = 1,
                            include_scene_info = True, mask_culling = False, save_scene=False):
        """ 
            Extract the point cloud from the dataset 
            Args:
                dataset: the dataset of the object
                    mask; 0 - background, 1, 2, 3... - object candidates
                add_transform: the additional transformation, apply on the first camera 
                    this is useful due to conflict of OpenGL and OpenCV Camera Convention
                skip_border_objects: whether to skip the border objects
                z_range: the range of the z-axis
                use_prot_comp: whether to use ProtoComp to complete the point cloud    
                downsample: the downsample rate of the point cloud
                include_scene_info: whether to include the scene information
                mask_culling: whether to use mask culling for each instance
                save_scene: whether to save the scene
            Returns:
                pc_full: the full point cloud
                pc_colors: the color of the point cloud
                comp_pcs:
                base_transform: the base transformation
                
        """
        fuse_pc_full = []
        fuse_pc_color = []
        fuse_pc_segments = {}
        pc_seg_world_dct = {}
        
        # take the first camera pose as the base transform
        if isinstance(dataset[0]['camtoworld'], torch.Tensor):
            base_transform = dataset[0]['camtoworld'].numpy() @ add_transform
        else:
            base_transform = dataset[0]['camtoworld'] @ add_transform
        
        masks = []
        w2cs = []
        Ks = []

        # fuse the point cloud of the object
        # views = [i for i in range(num_views)]
        num_views = len(dataset)
        for i in range(0, num_views):
            data_pack = dataset[i]
            segmap, rgb, depth, cam_K = data_pack['mask'], data_pack['image'], data_pack['depths'], data_pack['K']
            c2w = data_pack['camtoworld']
            if isinstance(c2w, torch.Tensor):
                c2w = c2w.numpy()

            H, W, C = rgb.shape
            depth = depth.reshape(H, W)
            if isinstance(segmap, torch.Tensor):
                segmap, rgb, depth, cam_K = segmap.numpy(), rgb.numpy(), depth.numpy(), cam_K.numpy()

            # print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                    skip_border_objects=skip_border_objects, 
                                                                        z_range=z_range, downsample=downsample)
            
            if not include_scene_info:
                del pc_segments[0]
            
            # it seems that Contact GraspNet is not SE(3)-Equivariant ... 
            rel = np.linalg.inv(base_transform) @ c2w
            pc_full = pc_full @ rel[:3, :3].T + rel[:3, 3:4].T
            for k, v in pc_segments.items():
                cur_pc_world = v @ c2w[:3, :3].T + c2w[:3, 3:4].T
                if k in pc_seg_world_dct:
                    pc_seg_world_dct[k].append(cur_pc_world)
                else:
                    pc_seg_world_dct[k] = [cur_pc_world]
            
            for k, v in pc_segments.items():
                pc_segments[k] = v @ rel[:3, :3].T + rel[:3, 3:4].T
                
            Ks.append(cam_K)
            w2cs.append(np.linalg.inv(rel))
            masks.append(segmap)

            fuse_pc_full.append(pc_full)
            fuse_pc_color.append(pc_colors)
            for k, v in pc_segments.items():
                if k not in fuse_pc_segments:
                    fuse_pc_segments[k] = []
                fuse_pc_segments[k].append(v)

        # concatenate point cloud from different view points
        for k, v in fuse_pc_segments.items():
            fuse_pc_segments[k] = np.concatenate(v, axis=0)
            # Cull all the masks
            if mask_culling:
                fuse_pc_segments[k] = mask_culling(fuse_pc_segments[k], masks, w2cs, Ks)
        
        pc_full = np.concatenate(fuse_pc_full, axis=0)
        pc_colors = np.concatenate(fuse_pc_color, axis=0)
        # pc_full, pc_colors = mask_culling(pc_full, masks, w2cs, Ks, pc_colors)
        
        # Run ProtoComp to complete the point cloud
        if use_prot_comp:
            from proto_comp.pc_infer import ProtoCompInference # type: ignore
            prot_comp_model = ProtoCompInference()
            comp_pcs = {}
            for k, pc_seg_lst in pc_seg_world_dct.items():
                comp_pcs[k] = []
                for cur_partial_pc in pc_seg_lst:
                    cur_full_pc = prot_comp_model(cur_partial_pc, "a box")
                    rel = np.linalg.inv(base_transform)
                    comp_pcs[k].append(cur_full_pc @ rel[:3, :3].T + rel[:3, 3:4].T)
                comp_pcs[k] = np.concatenate(comp_pcs[k], axis=0)
        else:
            comp_pcs = fuse_pc_segments
            
        # save all the points as scene information; 
        if save_scene and 'root_dir' in dataset[0]:
            # Save all the points in the world coordinate
            surr = fuse_pc_segments[0]
            obj_mean = np.mean(fuse_pc_segments[1], axis=0)
            obj_extent = np.max(
                np.max(fuse_pc_segments[1], axis=0) - np.min(fuse_pc_segments[1], axis=0)
            )
            surr = surr[np.linalg.norm(surr - obj_mean[None, :], axis=1) < 3 * obj_extent]
            
            if len(surr) > 2048:
                surr_pcd = o3d.geometry.PointCloud(); surr_pcd.points = o3d.utility.Vector3dVector(surr) # type: ignore
                surr_pcd = surr_pcd.farthest_point_down_sample(2048); surr_points = np.asarray(surr_pcd.points)
            else:
                surr_points = surr
            
            if len(fuse_pc_segments[1]) > 2048:
                object_pcd = o3d.geometry.PointCloud(); object_pcd.points = o3d.utility.Vector3dVector(fuse_pc_segments[1]) # type: ignore
                object_pcd = object_pcd.farthest_point_down_sample(2048); object_points = np.asarray(object_pcd.points)
            else:
                object_points = fuse_pc_segments[1]
                
            # segment_pcd = o3d.geometry.PointCloud()
            # segment_pcd.points = o3d.utility.Vector3dVector(fuse_pc_segments[1])
            # o3d.visualization.draw_geometries([segment_pcd])
        
            np.savez(
                os.path.join(dataset[0]['root_dir'], 'object.npz'),
                scene_pts = np.concatenate([surr_points, object_points]) @ base_transform[:3, :3].T + base_transform[:3, 3:4].T,
                target_index = np.concatenate([np.zeros(len(surr_points)), np.ones(len(object_points))])
            )

        
        print("Point Cloud Shape: ", pc_full.shape)
        print('Generating Grasps...')
        return pc_full, pc_colors, comp_pcs, base_transform

def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates 
      
      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            # _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pc)
            o3d_pcd = o3d_pcd.farthest_point_down_sample(npoints)
            pc = np.asarray(o3d_pcd.points)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
            pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False, convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)  

    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points

    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})

    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()
    
    if convert_to_internal_coords:
        pc[:,:2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc

class GraspEstimator:
    def __init__(self):
        pass

    @abstractmethod
    def predict_grasp_pose(self, **obs):
        """ 
            Predict the grasp pose of the object
                Returned poses are sorted by the confidence score
        
        Args:
            obs: the observation of the object
                dict: has keys 'data_path', 'viz'
        Returns:
            poses: [List] the predicted grasp poses
            meta: dict for additional information
        """
        pass

    @abstractmethod
    def step(self, **obs):
        pass