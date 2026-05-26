"""
Run Vision Foundation Models
"""
import argparse
import os
import os.path as osp
from typing import List, OrderedDict, Union

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
# depth anything V2
from gaussian_splatting_py.foundation.depth_anything_v2 import (
    DepthAnythingV2, create_model)
from gaussian_splatting_py.tools import depth_check
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_video_predictor import SAM2VideoPredictor
from scipy.optimize import lsq_linear

DP_MODEL_PATH = "/home/user/Documents/ws/ActiveTouch/src/gaussian_splatting/weights/depth_anything_v2_vitl.pth"
SAM2_PATH = "/home/user/segment-anything-2/checkpoints/sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"

# use bfloat16
# Disable this
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def learn_scale_and_offset_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = x
    return scale, offset


def learn_scale_and_offset_raw_scale_pos(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    # Use lsq_linear to enforce non-negative scale
    result = lsq_linear(A, b, bounds=([0, -np.inf], [np.inf, np.inf]))
    
    scale, offset = result.x
    return scale, offset

def learn_scale_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    # Ensure there are valid values and avoid division by zero
    if np.sum(dense_depth_valid ** 2) == 0 or len(dense_depth_valid) == 0:
        scale = 0
    else:
        # Compute the scale factor directly without an offset term
        scale = np.sum(sparse_depth_valid * dense_depth_valid) / np.sum(dense_depth_valid ** 2)
    
    offset = 0  # Offset is enforced to be zero
    return scale, offset

def convert_intrinsics(img, old_intrinsics = (360.01, 360.01, 243.87, 137.92), 
                            new_intrinsics = (1297.67, 1298.63, 620.91, 238.28), 
                            new_size=(1280, 720)):
    """
    Convert a set of images to a different set of camera intrinsics.
    Parameters:
    - images: List of input images.
    - old_intrinsics: Tuple (fx, fy, cx, cy) of the old camera intrinsics.
    - new_intrinsics: Tuple (fx, fy, cx, cy) of the new camera intrinsics.
    - new_size: Tuple (width, height) defining the size of the output images.
    Returns:
    - List of images converted to the new camera intrinsics.
    """
    old_fx, old_fy, old_cx, old_cy = old_intrinsics
    new_fx, new_fy, new_cx, new_cy = new_intrinsics
    width, height = new_size
    
    # Constructing the old and new intrinsics matrices
    K_old = np.array([[old_fx, 0, old_cx], [0, old_fy, old_cy], [0, 0, 1]])
    K_new = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])
    # Compute the inverse of the new intrinsics matrix for remapping
    K_new_inv = np.linalg.inv(K_new)
    
    # Construct a grid of points representing the new image coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T
    
    # Convert to the old image coordinates
    old_coords = K_old @ K_new_inv @ homogenous_coords
    old_coords /= old_coords[2, :]  # Normalize to make homogeneous
    
    # Reshape for remapping
    map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
    map_y = old_coords[1, :].reshape(height, width).astype(np.float32)
    
    # Remap the image to the new intrinsics
    converted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return converted_img

def warp_image(image, K, R, t):
    """
    Warp an image from the perspective of camera 1 to camera 2.

    :param image: Input image from camera 1
    :param K: Intrinsic matrix of both cameras
    :param R: Rotation matrix from camera 1 to camera 2
    :param t: Translation vector from camera 1 to camera 2
    :return: Warped image as seen from camera 2
    """
    # Compute the homography matrix
    H = compute_homography(K, R, t)

    # Warp the image using the homography
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image


def compute_homography(K, R, t):
    """
    Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K_inv = np.linalg.inv(K)
    H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
    return H

class DepthPredictionModule:
    def __init__(self, depth_encoder, depth_model_path):
        self.model:DepthAnythingV2 = create_model(depth_encoder, depth_model_path)
        print("Depth Prediction Module initialized.")

    def predict_depth(self, color_np: np.ndarray) -> np.ndarray:
        """ Predict the depth of the image. """
        monodepth_inverse = self.model.infer_image(color_np)

        # since the prediction is in inverse depth
        monodepth = 1.0 / np.clip(monodepth_inverse, a_min = 1e-6, a_max = 1e6)

        # we keep 0 as 0, since it is easy to filter out invalid depth
        # monodepth = np.where(monodepth_inverse > 0, monodepth, 0)
        monodepth = np.clip(monodepth, a_min=0, a_max=10)
        
        return monodepth_inverse, monodepth

    def align_depth(self, depth: np.ndarray, predicted_depth: np.ndarray,
                    color_intrinsics: np.ndarray, 
                    rgb: np.ndarray, warp_depth_frame:bool = True, use_sam: bool = False) -> np.ndarray:
        """ Align the predicted depth to the real depth. 
        
        Args:
            depth: The real depth image.
            predicted_depth: The predicted depth image.
            color_intrinsics: The intrinsics of the color camera.
            rgb: The color image.
            use_sam: If True, use SAM2 for semantic alignment.
        Returns:
            rs_depth: The depth image aligned to the color image.
            depth_np: The aligned depth
        """
        if warp_depth_frame:
            # convert instrinsics
            new_intrinsics_tup = (color_intrinsics[0], color_intrinsics[4], color_intrinsics[2], color_intrinsics[5])
            color_h, color_w = rgb.shape[:2]
            depth = convert_intrinsics(depth, 
                                    new_size=(color_w, color_h), 
                                    new_intrinsics=new_intrinsics_tup)
            
            # # depth to color transform
            # cmap = mpl.cm.get_cmap('plasma')
            # depth_norm = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            # depth_color = cmap(depth_norm)[:, :, :3]
            # plt.figure(figsize=(10, 10))
            # plt.imshow(depth_color)
            # plt.axis('off')
            # plt.savefig("depth.png")

            # TODO Depth 2 Color Conversion
            transform = np.array([
                [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, -0.01],
                [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.09],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -2.22044605e-16],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ])
            
            K = np.array([
                [360.01333,    0.      , 243.87228],
                [  0.      , 360.0133667, 137.9218444],
                [  0.      ,   0.      ,   1.      ]
            ])

            # warp the depth image
            rs_depth = warp_image(depth, K, transform[:3, :3], transform[:3, 3])
        else:
            rs_depth = depth
        
        scale, offset = learn_scale_and_offset_raw(predicted_depth, rs_depth)
        depth_np = (scale * predicted_depth) + offset
        depth_np[depth_np < 0] = 0

        return rs_depth, depth_np

class SAM2(SAM2AutomaticMaskGenerator):
    def __init__(self, points_per_side=64, pred_iou_thresh=0.95, use_m2m=False, SAM2_PATH=SAM2_PATH, MODEL_CFG=MODEL_CFG):
        sam2 = build_sam2(MODEL_CFG, SAM2_PATH, device ='cuda', apply_postprocessing=False)
        # build sam2 with points and iou threshold
        super().__init__(sam2, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, use_m2m=use_m2m)
        print("SAM2 initialized.")

    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def predict_masks(self, img: str) -> list:
        masks = super().generate(img)
        return masks

    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def generate(self, image: np.ndarray, mde_depth: np.ndarray, real_depth: np.ndarray,
                 is_challenge_object: bool = False, object_mask_path=None,
                 original_mde_depth_path: str = None) -> Union[None, list]:
        """
        Generate the masks for the image.

        Args:
            image: The input image.
            mde_depth: The depth image from monocular depth estimation.
            real_depth: The real depth image.
            mask_object: If not None, the object to mask and include in the background table segmentation

        Returns:
            Aligned depth image.
        """
        # get difference between the two depths' filtered 0s
        sparse_mask = real_depth > 0
        diff = np.mean(np.abs(mde_depth[sparse_mask] - real_depth[sparse_mask]))
        print(f"Diff: {diff}")
        
        # get the masks
        results = super().generate(image)
        all_masks = []
        
        background_mask = np.ones_like(image)
        background_mask = background_mask[:, :, 0]

        stats = np.zeros((mde_depth.shape[0], mde_depth.shape[1], 3), dtype=np.float32)
        # calculate the scale and offset for each mask region
        for result in results:
            # convert mask to 1 and 0
            bool_mask = result['segmentation']
            mask = bool_mask.astype(np.uint8)
            
            # reshape mask to image size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # perform alignment on mask
            mde_mask = mde_depth[bool_mask]
            real_mask = real_depth[bool_mask]
            scale, offset = learn_scale_and_offset_raw(mde_mask, real_mask)

            # compute pearson coefficient
            pearson = np.corrcoef(mde_mask, real_mask)[0, 1]
            
            # update mde
            # aligned_depth[bool_mask] = mde_depth[bool_mask] * scale + offset
            # remove negative values
            # aligned_depth[aligned_depth < 0] = 0
            
            # store the scale & offset in that region
            stats[bool_mask, 0] = pearson
            stats[bool_mask, 1] = scale
            stats[bool_mask, 2] = offset
            background_mask = background_mask * (1 - mask)
        
        # calculate scale and offset for the background
        background_mask = background_mask.astype(bool)
        mde_mask = mde_depth[background_mask]
        real_mask = real_depth[background_mask]
        scale, offset = learn_scale_and_offset_raw(mde_mask, real_mask)
        pearson = np.corrcoef(mde_mask, real_mask)[0, 1]
        stats[background_mask, 0] = pearson
        stats[background_mask, 1] = scale
        stats[background_mask, 2] = offset

        # align the depth
        aligned_depth = mde_depth * stats[:, :, 1] + stats[:, :, 2]
        aligned_depth[aligned_depth < 0] = 0

        # TODO Another way is to use the parameter with the highest pearson coefficient
        # pearson = stats[:, :, 0]
        # max_index_y, max_index_x = np.where(pearson >= (np.max(pearson) - 0.01) ) # avoid numerical issues
        # best_scale, best_offset = stats[max_index_y[0], max_index_x[0], 1], stats[max_index_y[0], max_index_x[0], 2]
        # aligned_depth = mde_depth * best_scale + best_offset
        # aligned_depth[aligned_depth < 0] = 0
        # aligned_depth[aligned_depth > 3] = 0
        
        if is_challenge_object:
            # read in old mde depth
            old_mde_depth = cv2.imread(original_mde_depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
            old_mde_mask = old_mde_depth[background_mask]
            
            scale, offset = learn_scale_and_offset_raw(old_mde_mask, real_mask)
            mask = cv2.imread(object_mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(bool)
            mde_depth[mask] = old_mde_depth[mask] * scale + offset
            mde_depth[mde_depth < 0] = 0
            
        if object_mask_path is not None and not is_challenge_object:
            mask = cv2.imread(object_mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(bool)
            old_mde_depth = cv2.imread(original_mde_depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
            
            # get median point of the mask
            mask = mask.astype(bool)
            old_mde_mask = old_mde_depth[mask]
            real_mask = real_depth[mask]
            scale, offset = learn_scale_and_offset_raw_scale_pos(old_mde_mask, real_mask)
            mde_depth[mask] = old_mde_depth[mask] * scale + offset
            mde_depth[mde_depth < 0] = 0
        
        # set values of sparse mask to 0 in img_diff
        diff = np.mean(np.abs(aligned_depth[sparse_mask] - real_depth[sparse_mask]))
        img_diff = np.abs(aligned_depth - real_depth)
        img_diff[~sparse_mask] = 0
        
        print(f"Diff: {diff}")
        print("Masks generated.")
        
        return all_masks, stats, aligned_depth

def preprocess(mde_depth):
        # Transform to convert image to tensor and normalize
        cmap = mpl.colormaps.get_cmap('Spectral_r')
        mde_depth_norm = (mde_depth - np.min(mde_depth)) / (np.max(mde_depth) - np.min(mde_depth))
        mde_depth_color = cmap(mde_depth_norm)[:, :, :3]
        return mde_depth_color

def depth_task(args):
    # build 
    model = DepthPredictionModule("vitl", DP_MODEL_PATH)

    # read color image
    color_np = cv2.imread(args.img_path)
    mde_inv, mde_depth = model.predict_depth(color_np)

    # visualize the depth
    mde_inv_color = preprocess(mde_inv)
    mde_depth_color = preprocess(mde_depth)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(mde_inv_color)
    axes[0].axis('off')
    axes[0].set_title("Monocular Depth Estimation (Inverse Depth)")
    axes[1].imshow(mde_depth_color)
    axes[1].axis('off')
    axes[1].set_title("Monocular Depth Estimation (Depth)")
    fig.savefig("mde_depth.png")

    # read real depth
    real_depth = cv2.imread(args.real_depth, cv2.IMREAD_UNCHANGED) / 1000.0

    # align the depth
    color_K = np.array([1297.672904, 0., 620.914026, 0., 1298.631344, 238.280325, 0., 0., 1.])
    warp_depth, aligned_depth = model.align_depth(real_depth, mde_depth, color_K, color_np)

    # visualize the aligned depth
    real_depth_color = preprocess(real_depth)
    warp_depth_color = preprocess(warp_depth)
    aligned_depth_color = preprocess(aligned_depth)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(real_depth_color)
    axes[0].axis('off')
    axes[0].set_title("Real Depth")
    axes[1].imshow(warp_depth_color)
    axes[1].axis('off')
    axes[1].set_title("Warped Depth")
    axes[2].imshow(aligned_depth_color)
    axes[2].axis('off')
    axes[2].set_title("Aligned Depth")
    fig.savefig("aligned_depth.png")

def depth_mask(args):
    dp_model = DepthPredictionModule("vitl", DP_MODEL_PATH)
    Sam2 = SAM2()
    
    color_np = cv2.imread(args.img_path)
    mde_inverse, mde_depth_pred = dp_model.predict_depth(color_np)
    real_depth = cv2.imread(args.real_depth, cv2.IMREAD_UNCHANGED) / 1000.0

    # align the depth
    color_K = np.array([1297.672904, 0., 620.914026, 0., 1298.631344, 238.280325, 0., 0., 1.])
    warp_depth, aligned_depth = dp_model.align_depth(real_depth, mde_depth_pred, color_K, color_np)

    # conver to rgb
    rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
    _, stats, mde_depth = Sam2.generate(rgb, mde_depth_pred, warp_depth)

    import pdb; pdb.set_trace()
    K = color_K.reshape(3, 3)
    mde_depth = mde_depth.astype(np.float32)
    _, _, pcd = depth_check.convert_rgbd_to_pcd(rgb, mde_depth, K, np.eye(4))

    cmap = mpl.colormaps.get_cmap('plasma')
    mde_depth_norm = (mde_depth - np.min(mde_depth)) / (np.max(mde_depth) - np.min(mde_depth))
    mde_depth_color = cmap(mde_depth_norm)[:, :, :3]

    plt.figure(figsize=(10, 10))
    plt.imshow(mde_depth_color)
    plt.axis('off')
    plt.savefig("mde_depth.png")
    plt.close()

    plt.figure()
    plt.imshow(stats[:, :, 0], cmap='plasma')
    plt.axis('off')
    plt.show()
    plt.savefig("pearson.png")
    plt.close()

    o3d.visualization.draw_geometries([pcd])

def sam2_task(args):
    Sam2 = SAM2()

    img = np.asarray(Image.open(args.img_path))
    masks = Sam2.predict_masks(img)
    
    # plot masks for SAM
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for mask in masks:
        seg_mask = mask['segmentation'].astype(np.uint8)
        show_mask(seg_mask, ax, random_color=True)

    fig.savefig("sam2_masks.png")

def update_monodepth(args):
    dp_model = DepthPredictionModule("vitl", DP_MODEL_PATH)
    Sam2 = SAM2()

    img_folder = '/'.join(args.img_path.split("/")[:-1])
    depth_folder = '/'.join(args.real_depth.split("/")[:-1])
    monofolder = img_folder.replace("images", "monodepth_new")
    monofolder_vis = img_folder.replace("images", "monodepth_new_vis")

    os.makedirs(monofolder, exist_ok=True)
    os.makedirs(monofolder_vis, exist_ok=True)

    img_names = os.listdir(img_folder)
    img_names.sort()

    depth_names = os.listdir(depth_folder)
    depth_names.sort()

    for img_name, depth_name in zip(img_names, depth_names):
        color_np = cv2.imread(osp.join(img_folder, img_name))
        mde_inverse, mde_depth = dp_model.predict_depth(color_np)

        real_depth = cv2.imread(osp.join(depth_folder, depth_name), cv2.IMREAD_UNCHANGED) / 1000.0
        # align the depth
        color_K = np.array([1297.672904, 0., 620.914026, 0., 1298.631344, 238.280325, 0., 0., 1.])
        warp_depth, aligned_depth = dp_model.align_depth(real_depth, mde_depth, color_K, color_np)

        # conver to rgb
        rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        all_masks, stats, mde_depth = Sam2.generate(rgb, mde_depth, warp_depth)

        # save the aligned depth
        mde_depth_u16 = (mde_depth * 1000).astype(np.uint16)
        cv2.imwrite(osp.join(monofolder, img_name), mde_depth_u16)

        # save the aligned depth visualization
        mde_color = preprocess(mde_depth) 
        plt.figure()
        plt.imshow(mde_color)
        plt.savefig(osp.join(monofolder_vis, img_name))
        plt.close()

        cmap = mpl.colormaps.get_cmap('plasma')
        pearson = (stats[:, :, 0] + 1 ) / 2
        pearson_color = cmap(pearson)[:, :, :3]

        plt.figure()
        plt.imshow(pearson_color)
        plt.savefig(osp.join(monofolder_vis, img_name.replace(".png", "_pearson.png")))
        plt.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image with depth information.")
    parser.add_argument('--img_path', type=str, help="Path to the image file", required=True)
    parser.add_argument('--real_depth', type=str, help="Path to the real depth file", required=True)
    parser.add_argument('--task', type=str, required=True, help="Task to perform")
    
    args = parser.parse_args()

    if args.task == "depth":
        depth_task(args)
    elif args.task == "sam2":
        sam2_task(args)
    elif args.task == "depth_mask":
        depth_mask(args)
    elif args.task == "update_monodepth":
        update_monodepth(args)
    else:
        print("Task not recognized.")
        exit(1)