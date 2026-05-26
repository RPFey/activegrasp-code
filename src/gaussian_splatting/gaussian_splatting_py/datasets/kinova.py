import argparse
import json as js
import logging
import os
from collections import namedtuple
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as SciR
from typing_extensions import Literal, assert_never

from gaussian_splatting_py.foundation.romatch_utils import (
    rgbd2pcd, romatch_triangulation, unproject_rgbd)

class KinovaParser(NamedTuple):
    points: np.ndarray
    points_rgb: np.ndarray
    camtoworlds: np.ndarray
    Ks_dict: Dict[str, np.ndarray]
    imsize_dict: Dict[str, Tuple[int, int]]
    bounds: np.array
    extconf: Dict[str, Any]


class KinovaDataset:
    """ 
    This is the class for loading the images collected by Kinova arm 
    The data folder should look like this:
    -- root_dir
        -- transforms.json
        -- images
            -- 000000.png
            -- 000001.png
    
    Functions:
        1. Load camera poses and images
        2. Run SAM2 & DepthAnythingV2 model
    """
    def __init__(self, root_dir:str = None, 
                 skip_monodepth:bool=False,
                 mount_mask:np.ndarray=None, 
                 touch_cfg:str="config/digit.yaml",
                 merge_touch:bool=False,
                  **kwargs) -> None:
        
        self.root_dir = root_dir

        self.poses = None
        self.K = None
        self.imgs = []
        self.masks = []
        self.depths = []
        self.mask_paths = []
        self.skip_monodepth = skip_monodepth
        self.mount_mask = mount_mask

        if touch_cfg == "config/digit.yaml":
            touch_cfg = "/root/ActiveTouch/src/gaussian_splatting/config/digit.yaml"
            logging.warning(f"Override touch_cfg to {touch_cfg}")
        
        if root_dir is not None:
            self.load_transforms(os.path.join(root_dir, "transforms.json"))
            if merge_touch:
                self.rgb_frame_idxs = self.rgb_frame_idxs + self.touch_frame_idxs
            assert len(self.poses) == len(self.imgs), "Number of poses and images should be the same"
        else:
            assert "K" in kwargs, "Camera instrinsics not provided"
            assert "poses" in kwargs, "Poses not provided"
            assert "imgs" in kwargs, "images not provided"
            
            self.K = kwargs["K"]
            self.poses = kwargs["poses"]
            self.imgs = kwargs["imgs"]
            self.valid_rgb_frames = [True] * len(self.imgs)

            # try to get depth and masks
            self.depths = kwargs.get("depths", [])
            self.mask_paths = kwargs.get("mask_paths", [])

            self.valid_depth_frames = [True] * len(self.imgs)
            self.touch_frame_idxs = []
            self.rgb_frame_idxs = list(range(len(self.imgs)))
            
            if self.mount_mask is not None:
                self.masks = [self.mount_mask] * len(self.imgs)
            else:
                self.masks = []
        
        try:
            with open(touch_cfg, "r") as f:
                self.touch_cfg = yaml.safe_load(f)
            if self.touch_cfg["sensor"]["resolution"].upper() == "QVGA":
                self.touch_P = np.array(self.touch_cfg["sensor"]["P"])
            else:
                assert_never(self.touch_cfg["sensor"]["resolution"])
            # 320x240

            K = np.eye(3)
            # transformation of focal length, not aboslute sure about the x100 factor but might be cm to m transformation etc.
            # but semes fine from visualiztion
            K[0, 0] = self.touch_P[0, 0] * 100
            K[1, 1] = self.touch_P[1, 1] * 100
            K[0, 2] = 240 / 2
            K[1, 2] = 320 / 2
            self.touch_K = K
        except Exception as e:
            print(f"laod fail cwd: {os.getcwd()}, fn: {touch_cfg}")
            print(f"Error loading touch config: {e}")
            self.touch_cfg = None
            self.touch_P = None
            self.touch_K = None
    
    def get_parser(self, init_type: Literal["sfm", "romatch", "rgbd"], triangulate_nums: int=3, frame_interval: int=7, init_from_touch: bool=False, add_semantics=False) -> KinovaParser:
        if add_semantics:
            assert init_type == "romatch", "Only romatch supports add_semantics"
        if init_type == "romatch":
            xyzs, rgbs = romatch_triangulation(self, triangulate_nums=triangulate_nums, add_semantics=add_semantics)
        elif init_type == "rgbd":
            xyzs, rgbs = unproject_rgbd(self, frame_interval=frame_interval)
            # Unproject touch depth
        
        TOUCH_SKIP = 7
        if self.touch_P is not None and init_from_touch:
            extra_xyzs = []
            extra_rgbs = []
            for idx in self.touch_frame_idxs:
                depth = self.depths[idx]
                rgb = np.zeros((*depth.shape, 3), dtype=np.uint8) 
                # rgb[..., 0] = 255
                rgb[..., :] = 255 // 2

                c2w = self.poses[idx]
                xyz, rgb = rgbd2pcd(rgb, depth, self.touch_K, c2w)
                extra_xyzs.append(xyz[::TOUCH_SKIP])
                extra_rgbs.append(rgb[::TOUCH_SKIP])
            xyzs = np.concatenate([xyzs, *extra_xyzs], axis=0)
            rgbs = np.concatenate([rgbs, *extra_rgbs], axis=0)
        
        bounds = np.array([0.05, 3.0])

        return KinovaParser(points=xyzs, points_rgb=rgbs, camtoworlds=self.poses, 
                            Ks_dict={str(i): self.K for i in range(len(self))}, 
                            bounds=bounds,
                            extconf={'spiral_radius_scale': 0.3},
                            imsize_dict={str(i): img.shape[:2][::-1] for i, img in enumerate(self.imgs)}) # h, w -> w, h
    
    def load_transforms(self, file_name):
        """ Load the transforms.json file 
            The poses is c2w
        
        This function loads the poses, K, img_paths, depth_paths, mask_paths from the transforms.json file
        """
        with open(file_name, "r") as f:
            transforms = js.load(f)
        
        # load camera instrinsics
        fl_x = transforms["fl_x"]
        fl_y = transforms["fl_y"]
        cx = transforms["cx"]
        cy = transforms["cy"]
        self.K  = np.array([[fl_x, 0, cx],
                            [0, fl_y, cy],
                            [0, 0, 1]])
        
        # load camera poses
        valid_rgb_frames = []
        poses = []
        valid_depth_frames = []
        self.touch_frame_idxs = []
        self.rgb_frame_idxs = []

        logged = False
        for idx, frame in enumerate(transforms["frames"]):
            # if "touch_frame_path" in frame:
            #     continue
            c2w = np.array(frame["transform_matrix"])
            # c2w = c2w @ np.array([[1, 0, 0, 0],
            #                        [0, 0, -1, 0],
            #                        [0, 1, 0, 0],
            #                        [0, 0, 0, 1.]])
            # I comment out this line because if rs image path exits
            # then the c2w is already the rs2world pose
            poses.append(c2w)
            
            # if "rs_image_path" in frame: # need to transform from built-in to rs
            #     trans = np.array([0.00435657, -0.02899439, 0.03026354])
            #     q = np.array([0.0003816, 0.038657,  0.99923303, -0.00622467]) #rs -> builtin
            #     rot = SciR.from_quat(q).as_matrix()
            #     rs_to_builtin = np.eye(4)
            #     rs_to_builtin[:3,:3] = rot
            #     rs_to_builtin[:3,3] = trans
            #     c2w = np.dot(c2w, rs_to_builtin)
            
            valid_rgb_frame = True
            if "rs_image_path" in frame:
                img = imageio.imread(os.path.join(self.root_dir, frame["rs_image_path"]))[..., :3]
                img_source = "real sense"
            elif "file_path" in frame:
                img = imageio.imread(os.path.join(self.root_dir, frame["file_path"]))[..., :3]
                img_source = "kinova"
            else:
                img = imageio.imread(os.path.join(self.root_dir, frame["touch_frame_path"]))[..., :3]
                img_source = "touch"
                valid_rgb_frame = False
            
            if valid_rgb_frame:
                self.rgb_frame_idxs.append(idx)

            if not logged:
                logging.info(f"Image source: {img_source}")

            self.imgs.append(img)
            valid_rgb_frames.append(valid_rgb_frame)

            valid_depth_frame = True 
            depth_source = None
            # TODO depth file & Mask file 
            if not self.skip_monodepth and "monodepth_path" in frame:
                depth = cv2.imread(os.path.join(self.root_dir, frame["monodepth_path"]), cv2.IMREAD_UNCHANGED) / 1000.
                self.depths.append(depth)
                depth_source = "monodepth"
            elif "rs_depth_path" in frame:
                depth = cv2.imread(os.path.join(self.root_dir, frame["rs_depth_path"]), cv2.IMREAD_UNCHANGED) / 1000.
                self.depths.append(depth)
                depth_source = "real sense"
            elif "depth_path" in frame:
                depth = cv2.imread(os.path.join(self.root_dir, frame["depth_path"]), cv2.IMREAD_UNCHANGED) / 1000.
                self.depths.append(depth)
                depth_source = "built-in"
            elif "touch_depth_path" in frame:
                depth = cv2.imread(os.path.join(self.root_dir, frame["touch_depth_path"]), cv2.IMREAD_UNCHANGED) / 1e6
                # depth[:] = 0.02 / 1e2
                
                self.depths.append(depth)
                depth_source = "touch"
                valid_depth_frame = False # temporarily disable touch depth
                self.touch_frame_idxs.append(idx)
            
            if not logged:
                logging.info(f"Depth source: {depth_source}")
            
            valid_depth_frames.append(valid_depth_frame)

            # Masks
            if "mask" in frame:
                # mask = 1. - cv2.imread(os.path.join(self.root_dir, frame["mask"]), cv2.IMREAD_GRAYSCALE) / 255
                # Don't know why the mask was inverted
                mask = cv2.imread(os.path.join(self.root_dir, frame["mask"]), cv2.IMREAD_GRAYSCALE) / 255
                self.masks.append(mask)
            elif "touch_depth_path" in frame: # deal with touch
                mask = np.ones_like(depth)
                self.masks.append(mask)
            else:
                if self.mount_mask is not None:
                    self.masks.append(self.mount_mask) # we don't have mask during online training

            logged = True

        self.poses = np.stack(poses)
        self.valid_rgb_frames = np.array(valid_rgb_frames)
        self.valid_depth_frames = np.array(valid_depth_frames)

        # TODO Normalize the poses

    def __len__(self) -> int:
        """ Return the number of images """
        # return len(self.poses)
        return len(self.rgb_frame_idxs)

    def __getitem__(self, idx:int):
        """ Get the image, pose, depth, mask at index idx """
        idx = self.rgb_frame_idxs[idx]

        img = self.imgs[idx]
        pose = self.poses[idx]

        data = {
            "K": torch.from_numpy(self.K).float(),
            "camtoworld": torch.from_numpy(pose).float(),
            "image": torch.from_numpy(img).float(),
            "image_id": idx,  # the index of the image in the dataset
            "valid_rgb": self.valid_rgb_frames[idx],
            "valid_depth": self.valid_depth_frames[idx],
        }

        if len(self.depths) > 0:
            depth = self.depths[idx]
            H, W = depth.shape
            depth = torch.from_numpy(depth).float()

            meshgrid_y, meshgrid_x = torch.meshgrid([torch.arange(H), torch.arange(W)])
            meshgrid = torch.stack([meshgrid_x, meshgrid_y], dim=-1).float().reshape(-1, 2)
            points = depth.reshape(-1)

            data["points"] = meshgrid
            data["depths"] = points

        if len(self.masks) > 0:
            mask = self.masks[idx]
            mask = torch.from_numpy(mask).bool()
            data["mask"] = mask
        
        return data
    
    def export_colmap(self, datapath:str):
        """ Export the dataset to colmap format """
        assert self.K is not None, "Camera instrinsics not loaded"
        img = imageio.imread(os.path.join(self.root_dir, self.img_paths[0]))[..., :3]
        h, w = img.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # export Camera
        with open(os.path.join(datapath, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

        # export images
        with open(os.path.join(datapath, "images.txt"), "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            for idx, img_path in enumerate(self.img_paths):
                pose = self.poses[idx]
                pose = np.linalg.inv(pose)
                R = pose[:3, :3]
                t = pose[:3, 3]
                q = SciR.from_matrix(R).as_quat(scalar_first=True)
                name = img_path.split("/")[-1]
                f.write(f"{idx + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n")
                f.write("\n")

        # export points (empty here)
        with open(os.path.join(datapath, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, default="/home/user/Documents/data/2024-09-28-19-45-36")
    arg.add_argument("--output", type=str, default="/home/user/Documents/data/2024-09-28-19-45-36/colmap")
    args = arg.parse_args()

    dataset = KinovaDataset(args.data)

    # export the dataset to colmap format
    dataset.export_colmap(args.output)
