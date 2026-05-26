# hand eye calibration
# To collect data, you can run the moveit_controller node, 
# but change the transformation matrix in vision node from color_optical_frame to tool_frame
# so that the transformation matrix stored in the `transforms.json` is the transformation from the camera to the tool frame.
# Then, you can run the following script to calibrate the transformation from the camera to the tool frame.
#
# python3.11 cam2hand.py --imgs /path/to/images --hand /path/to/transforms.json
# 
# The script will output the transformation from the camera to the tool frame.

import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sciR
from lietorch import SE3, LieGroupParameter
import torch
import json as js

from gaussian_splatting_py.vision_utils import compute_pose_from_marker, calib_cam
from gaussian_splatting_py.vision_utils import CHECHERBOARD_SIZE, CHECHERBOARD_SIZE_IN_METER

def load_transforms(file_path):
    with open(file_path, "r") as f:
        data = js.load(f)
    
    frames = data["frames"]
    poses = []
    for f in frames:
        transform = f["transform_matrix"]
        transform = np.array(transform).reshape(4, 4)

        tvec = transform[:3, 3]
        quat  = sciR.from_matrix(transform[:3, :3]).as_quat()
        p = np.concatenate([tvec, quat])
        poses.append(p)

    poses = np.array(poses)
    return poses

def compute_residue(world2cam_se3, cam2hand_se3, world2base_se3, hand2base_se3):
    """
    Compute the residue for the optimization

    Args:
        world2cam_se3: SE3, world to camera transformation
        cam2hand_se3: SE3, camera to hand transformation
        world2base_se3: SE3, world to base transformation
        hand2base_se3: SE3, hand to base transformation
    Returns:
        res_vec: torch.Tensor, residue vector
    """

    world2cam2base = hand2base_se3 * cam2hand_se3[[0]] * world2cam_se3
    res =  world2base_se3[[0]].inv() * world2cam2base
    res_vec = res.log()
    return res_vec

def hand_eye_calib(opt):
    imgs1 = os.listdir(opt.imgs)
    imgs1_paths = [os.path.join(opt.imgs, img) for img in imgs1]
    imgs1_paths.sort()
    rvecs, tvecs = calib_cam(imgs1_paths, opt.default_intrinsic)

    # store in (x, y, z, qx, qy, qz, qw) format
    hand2base = load_transforms(opt.hand)

    hand2base_R = np.stack([sciR.from_quat(h[3:]).as_matrix() for h in hand2base], axis=0)
    hand2base_t = np.stack([h[:3] for h in hand2base], axis=0)

    world2cam_R = np.stack([sciR.from_rotvec(rvec).as_matrix() for rvec in rvecs], axis=0)
    world2cam_t = tvecs

    import pdb; pdb.set_trace()
    R, t = cv2.calibrateHandEye(hand2base_R, hand2base_t, world2cam_R, world2cam_t, cv2.CALIB_HAND_EYE_DANIILIDIS)
    
    quat = sciR.from_matrix(R).as_quat()
    print("Transformation from camera to hand frame:")
    print("Translation: ", t)
    print("Rotation: ", quat)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--imgs", type=str, required=True, help="Path to images from camera")
    args.add_argument("--hand", type=str, required=True, help="Path to tool frame")
    args.add_argument("--default_intrinsic", action="store_true", default=False, help="Use default intrinsic")
    opt = args.parse_args()

    hand_eye_calib(opt)

