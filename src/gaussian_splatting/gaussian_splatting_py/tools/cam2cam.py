# Cam 2 Cam calibration file
# Usage: put the images in folder /path/to/imgs1 and the 
# corresponding images in folder /path/to/imgs2
# Run the script with the following command:
# python cam2cam.py --imgs1 /path/to/imgs1 --imgs2 /path/to/imgs2
# to improve quality, take ~10 images, you don't need to ensure the entire board is in view.

import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sciR
from gaussian_splatting_py.vision_utils import compute_pose_from_marker, calib_cam, CHECHERBOARD_SIZE, CHECHERBOARD_SIZE_IN_METER

def calib_cam2cam(args):
    imgs1 = os.listdir(args.imgs1)
    imgs1_paths = [os.path.join(args.imgs1, img) for img in imgs1]
    imgs1_paths.sort()

    imgs2 = os.listdir(args.imgs2)
    imgs2_paths = [os.path.join(args.imgs2, img) for img in imgs2]
    imgs2_paths.sort()
    rvecs1, tvecs1 = calib_cam(imgs1_paths, use_default=args.default_intrinsic)
    rvecs2, tvecs2 = calib_cam(imgs2_paths, use_default=args.default_intrinsic)

    def get_tranform(rvecs, tvecs):
        transforms = []
        for rvec, tvec in zip(rvecs, tvecs):
            rmat = cv2.Rodrigues(rvec)[0]
            transform = np.eye(4)
            transform[:3, :3] = rmat
            transform[:3, 3] = tvec.squeeze()
            transforms.append(transform)
        return transforms
    
    # get the transforms for cam1
    cam1_transforms = get_tranform(rvecs1, tvecs1)
    # get the transforms for cam2
    cam2_transforms = get_tranform(rvecs2, tvecs2)

    rel_transforms = [transform1 @ np.linalg.inv(transform2) for transform1, transform2 in zip(cam1_transforms, cam2_transforms)]
    rel_tvecs = np.array([transform[:3, 3] for transform in rel_transforms])
    rel_quats = np.array([sciR.from_matrix(transform[:3, :3]).as_quat() for transform in rel_transforms])

    mean_tvecs = np.mean(rel_tvecs, axis=0)
    mean_quats = np.mean(rel_quats, axis=0)
    mean_quats /= np.linalg.norm(mean_quats)

    print("Mean relative translation: ")
    for i in range(3):
        print(f"{mean_tvecs[i]:.6f} ")
    print("Mean relative rotation: ")
    for i in range(4):
        print(f"{mean_quats[i]:.6f} ") 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--imgs1", type=str, required=True, help="Path to images from camera 1")
    args.add_argument("--imgs2", type=str, required=True, help="Path to images from camera 2")
    args.add_argument("--default_intrinsic", action="store_true", default=False, help="Use default intrinsic")
    args = args.parse_args()

    calib_cam2cam(args)
