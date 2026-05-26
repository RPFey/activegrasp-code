import open3d as o3d
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import json as js

class O3DTsdfIntegration:
    def __init__(self):
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.008,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def integrate(self, pcd, pose):
        self.tsdf_volume.integrate(pcd, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault), pose)
        return self.tsdf_volume.get_volume()

    def get_volume(self):
        return self.tsdf_volume.get_volume()
    
    def get_tsdfs(self, points):
        """ Get TSDF values at points 
        
        Args:
            points: Points to query, (N, 3)
        Returns:
            tsdfs: TSDF values at points, (N,)
        """
        tsdfs = self.tsdf_volume.tsdf_at(points) * self.tsdf_volume.sdf_trunc
        return tsdfs

    def get_mesh(self):
        return self.tsdf_volume.extract_triangle_mesh()
    
    def build_from_data_folder(self, gs_data_dir:str):
        # read json file
        with open(osp.join(gs_data_dir, 'transforms.json')) as f:
            data = js.load(f)

        frames = data["frames"]
        cam_w = data["w"]
        cam_h = data["h"]
        focal_x = data["fl_x"]
        focal_y = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]

        # create camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(cam_w, cam_h, focal_x, focal_y, cx, cy)

        for idx in tqdm(range(len(frames)), desc="Integrating TSDF"):
            frame =  frames[idx]
            color_key = "rs_image_path" if "rs_image_path" in frame else "file_path"
            color = cv2.imread(osp.join(gs_data_dir, frame[color_key]))
            # rgb to bgr
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            
            depth_key = "rs_depth_path" if "rs_depth_path" in frame else "depth_path"
            depth = cv2.imread(osp.join(gs_data_dir, frame[depth_key]), cv2.IMREAD_UNCHANGED)
            depth_meter = depth.astype(np.float32) / 1000.0

            # create camera pose
            c2w = np.array(frame["transform_matrix"])
            extrinsic = np.linalg.inv(c2w)

            # create rgbd image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color),
                o3d.geometry.Image(depth_meter),
                depth_scale=1.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False)
            
            # integrate rgbd image into TSDF volume
            self.tsdf_volume.integrate(rgbd, intrinsic, extrinsic)

    def get_tsdf_query_func(self):
        """ Get a function that queries TSDF values at points 
        
        Returns:
            function: A function that queries TSDF values
        """

        return lambda x: self.get_tsdfs(x)

if __name__ == "__main__":
    import argparse 

    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data")
    opt = args.parse_args()

    tsdf_integrator = O3DTsdfIntegration()
    tsdf_integrator.build_from_data_folder(opt.data_dir)

    mesh = tsdf_integrator.get_mesh()
    o3d.visualization.draw_geometries([mesh])