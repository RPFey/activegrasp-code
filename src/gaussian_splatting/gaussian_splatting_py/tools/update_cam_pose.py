# Cam 2 Cam calibration file
# Usage: put the images in folder /path/to/imgs1 and the 
# corresponding images in folder /path/to/imgs2
# Run the script with the following command:
# python cam2cam.py --imgs1 /path/to/imgs1 --imgs2 /path/to/imgs2

import cv2
import numpy as np
import argparse
import os
import json as js
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sciR
from gaussian_splatting_py.vision_utils import compute_pose_from_marker, CHECHERBOARD_SIZE, CHECHERBOARD_SIZE_IN_METER

def update_cam_pose(args):
    # load transforms
    file_name = os.path.join(args.data_dir, "transforms.json")
    with open(file_name, "r") as f:
            transforms = js.load(f)
        
    # load camera instrinsics
    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx = transforms["cx"]
    cy = transforms["cy"]
    K  = np.array([[fl_x, 0, cx],
                    [0, fl_y, cy],
                    [0, 0, 1]])
    
    import pdb; pdb.set_trace()
    # load camera poses
    for idx, frame in enumerate(transforms["frames"]):
        image_path = os.path.join(args.data_dir, frame["rs_image_path"])
        img = cv2.imread(image_path)

        rvet, rvec, tvec = compute_pose_from_marker(img, K, np.zeros((5,)), viz=True)
        pose = np.eye(4)
        cv2.Rodrigues(rvec, pose[:3, :3])
        pose[:3, 3] = tvec.flatten()
        pose = np.linalg.inv(pose)
        pose_list = [p.tolist() for p in pose]
        transforms["frames"][idx]["transform_matrix"] = pose_list

    # save the updated transforms
    updated_file_name = os.path.join(args.data_dir, "transforms_updated.json")
    with open(updated_file_name, "w") as f:
        js.dump(transforms, f)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, required=True, help="path to dataset")
    # args.add_argument("--default_intrinsic", action="store_true", default=False, help="Use default intrinsic")
    args = args.parse_args()

    update_cam_pose(args)
