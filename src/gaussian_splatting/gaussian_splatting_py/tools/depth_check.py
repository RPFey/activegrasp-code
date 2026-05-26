# Read Depth files and convert them to point cloud

import open3d as o3d
import cv2
import numpy as np
from gaussian_splatting_py.datasets import KinovaDataset
import argparse

def convert_rgbd_to_pcd(image, depth, intrinsics, c2w, 
                        depth_scale=1.0, depth_trunc=7.0):
    """
    Convert RGBD image to point cloud

    Args:
        image: np.array (H, W, 3) np.uint8
        depth: np.array (H, W) np.uint16
        intrinsics: np.array (3, 3) np.float32
        extrinsics: np.array (4, 4) np.float32

    Returns:
        points: np.array (N, 3) np.float32
        colors: np.array (N, 3) np.float32
        pcd: o3d.geometry.PointCloud
    """
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(image),
        o3d.geometry.Image(depth),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image.shape[1], image.shape[0], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    pcd.transform(c2w)
    
    # read colors & points
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)

    return points, colors, pcd

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="kinova")
    opt = args.parse_args()

    dataset = KinovaDataset(opt.dataset)
    
    num_images_to_read = 3
    frame_interval = 10
    pcds = []
    for i in range(0, len(dataset), frame_interval):
        pack = dataset[i]

        K = pack["K"].numpy()
        c2w = pack["camtoworld"].numpy()
        img = pack["image"].numpy().astype(np.uint8)
        depth = pack["depths"].numpy()
        depth = np.ascontiguousarray(depth.reshape(img.shape[:2]))

        _, _, pcd = convert_rgbd_to_pcd(img, depth, K, c2w)
        pcds.append(pcd)
    
    # o3d.visualization.draw_plotly(pcds)
    o3d.visualization.draw_geometries(pcds)

