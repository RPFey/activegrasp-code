"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import os
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as sciR

import numpy as np
import tyro
import open3d as o3d
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

from gaussian_splatting_py.datasets.colmap import Dataset, Parser
import matplotlib.pyplot as plt
import matplotlib as mpl

import gaussian_splatting_py.datasets as datasets
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

def equal_hist(uncern):
    H, W = uncern.shape

    # Histogram equalization for visualization
    uncern = uncern.flatten()
    median = np.median(uncern)
    bins = np.append(np.linspace(uncern.min(), median, len(uncern)), 
                            np.linspace(median, uncern.max(), len(uncern)))
    # Do histogram equalization on uncern  
    # bins = np.linspace(uncern.min(), uncern.max(), len(uncern) // 20)
    hist, bins2 = np.histogram(uncern, bins=bins)
    # Compute CDF from histogram
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf = np.hstack(([0], cdf))
    cdf = cdf / cdf[-1]
    # Do equalization
    binnum = np.digitize(uncern, bins, True) - 1
    neg = np.where(binnum < 0)
    binnum[neg] = 0
    uncern_aeq = cdf[binnum] * bins[-1]

    uncern_aeq = uncern_aeq.reshape(H, W)
    uncern_aeq = (uncern_aeq - uncern_aeq.min()) / (uncern_aeq.max() - uncern_aeq.min())
    return uncern_aeq 

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

def main(
    data_path: Path,
    /,
    pcd_frame_every: int = 4,
    downsample_factor: int = 4,
    share: bool = False,
    port: int = 9090,
) -> None:
    server = viser.ViserServer(port=port, show_axis=False)
    if share:
        server.request_share_url()

    print("Loading frames!")
    if os.path.exists(f"{data_path}/transforms.json"):
        # load the Kinova Collected Dataset
        print(f"loading kinova dataset with {data_path}/transforms.json")
        trainset = datasets.KinovaDataset(data_path, merge_touch=True, skip_monodepth=True)
        parser = trainset.get_parser("rgbd", frame_interval=10, init_from_touch=False)
    else:
        parser = Parser(
                    data_dir=data_path,
                    factor=4,
                    normalize=True,
                    test_every=8,
                )
        trainset = Dataset(
                parser,
                split="train",
                patch_size=None,
                load_depths=False,
            )

    gui_point_size = server.gui.add_slider(
        "Point size",
        min=0.0001,
        max=0.02,
        step=1e-4,
        initial_value=0.001,
    )

    gui_show_world_axes = server.gui.add_checkbox("Show world axes", initial_value=False)

    # Load in frames.
    base_frame = server.scene.add_frame(
        "/frames",
        # wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=gui_show_world_axes.value,
    )
    frame_nodes: list[viser.FrameHandle] = []

    scene_pcd = o3d.geometry.PointCloud()
    for i, data in enumerate(tqdm(trainset)):
        # Add base frame.
        # if not data.get("valid_depth", True):
        #     continue
        # TODO - Skip the last one, since it 
        # is the recent captured view

        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        rgb = data["image"].numpy()
        K = data["K"].numpy()
        camtoworld = data["camtoworld"].numpy()

        img = data["image"].numpy().astype(np.uint8)
        depth = data["depths"].numpy()
        depth = np.ascontiguousarray(depth.reshape(img.shape[:2]))
        _, _, pcd = convert_rgbd_to_pcd(img, depth, K, camtoworld)
        scene_pcd += pcd
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=0.001)
        
        if i > 3:
            continue

        # Place the frustum.
        fov = 2 * np.arctan2(rgb.shape[0] / 2, K[0, 0])
        aspect = rgb.shape[1] / rgb.shape[0]
        img_size = 0.05
        quat = sciR.from_matrix(camtoworld[:3, :3]).as_quat(scalar_first=True)
        
        frustum = server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            color=(0, 0, 0),
            aspect=aspect,
            scale=img_size / 2,
            wxyz=quat,
            # line_width=8,
            position=camtoworld[:3, 3],
        )

        # Add some axes.
        axes = server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=img_size / 2,
            axes_radius=img_size / 20,
        )

        img_node = server.scene.add_image(
            f"/frames/t{i}/frustum/axes",
            render_height=img_size,
            render_width=img_size * aspect,
            image=rgb[::downsample_factor, ::downsample_factor] / 255.,
        )
        
    point_node = server.scene.add_point_cloud(
        name=f"/frames/point_cloud",
        points=np.asarray(scene_pcd.points),
        colors=np.asarray(scene_pcd.colors),
        point_size=gui_point_size.value,
        point_shape="rounded",
    )
    
    # load pose evaluation results
    # 5 - training set size is 5.
    methods = {
        "FisherGrasp": "/home/leiboshu/ActiveGrasp/conformal_data_cu/se3diff_dual_FisherGrasp_H_lambda1e-3/s7-ep3/iter1/2025-09-17-17-56-03-s7-ep3",
        # "ActiveNGF": "/home/leiboshu/ActiveGrasp/f2a2/se3diff_dual_ActiveNGF_score_grasp-bkgd1.5/s0-ep5/iter0/2025-09-17-03-38-15-s0-ep5",
        # "ACE": "/root/ActiveGrasp/exp/se3diff_scene_ACE/2025-05-09-00-45-29-s5-ep0",
        # "random": "/root/ActiveGrasp/exp/se3diff_scene_ActiveNGF/2025-05-08-19-40-34-s5-ep0",
    }
    
    Colors = {
        "FisherGrasp": [0.0, 1.0, 0.0],
        # "random": [1.0, 0.0, 0.0],
        # "ACE": [0.0, 0.0, 1.0],
        "ActiveNGF": [1.0, 1.0, 0.0],
    }
    
    for name, filename in methods.items():
        # load the pose evaluation results
        pose_eval_f = os.path.join(data_path, f"{filename}/pose_eval_3.npz")
        if os.path.exists(pose_eval_f):
            f_ = np.load(pose_eval_f)
            poses = f_["poses"]
            scores = f_["scores"]

            # get color map - plsama
            cmap = mpl.cm.plasma
            scores = equal_hist(scores[None, :])[0]
            
            score_sorted_idxs = np.argsort(scores)[::-1]
            score_sorted_idxs = score_sorted_idxs # [:3]
            for i, (pose, score) in enumerate(zip(poses[score_sorted_idxs], scores[score_sorted_idxs])):
                # color = Colors[name]
                color = cmap(score)[:3]
                quat = sciR.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
                
                # i = score_sorted_idxs[i]
                frame_nodes.append(server.scene.add_frame(f"/frames/{name}{i}", show_axes=False, visible=True))
                server.scene.add_camera_frustum(
                    f"/frames/{name}{i}/frustum",
                    color=[int(255 * c) for c in color],
                    fov=fov,
                    aspect=aspect,
                    scale=img_size / 2,
                    # line_width=8,
                    wxyz=quat,
                    position=pose[:3, 3],
                )
                # Add some axes.
                axes = server.scene.add_frame(
                    f"/frames/{name}{i}/frustum/axes",
                    axes_length=img_size / 2,
                    axes_radius=img_size / 20,
                )
    
    # Frame step buttons.
    # GUI is a bit buggy right now
    # @gui_point_size.on_update
    # def _(_) -> None:
    #     # point_node.point_size = gui_point_size.value
    #     point_node = server.scene.add_point_cloud(
    #         name=f"/frames/point_cloud",
    #         points=parser.points,
    #         colors=parser.points_rgb,
    #         point_size=gui_point_size.value,
    #         point_shape="rounded",
    #     )
    #     server.flush()
    
    # @gui_show_world_axes.on_update
    # def _(_) -> None:
    #     base_frame.show_axes = gui_show_world_axes.value
    #     print(f"show world axes: {gui_show_world_axes.value}")
    #     server.flush()
    
    while True:
        time.sleep(1 / 30.)


if __name__ == "__main__":
    tyro.cli(main)