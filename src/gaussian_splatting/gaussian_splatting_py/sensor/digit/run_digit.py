# Digit Interface Class For Active Touch Project
# Usage 
# Capture camera images
# python run_depth.py --device /dev/video2 --model /path/to/weight --background /path/to/background --task capture
# Create point cloud from images
# python run_depth.py --model /path/to/weight --background /path/to/background --task point_cloud

import logging
import typing

import cv2
import pandas as pd
import numpy as np
import torch
import copy
import yaml
import os
import open3d as o3d
import torch.nn as nn
import torch.nn.functional as F
import gaussian_splatting_py.digit.geom_utils as geom_utils
from tqdm.auto import tqdm
from glob import glob


logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preproc_mlp(image) -> torch.Tensor:
    """ Preprocess image for input to model.

    Args: image: OpenCV image in BGR format
    Return: tensor of shape (R*C,5) where R=320 and C=240 for DIGIT images
    5-columns are: X,Y,R,G,B

    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    xy_coords = np.flip(np.column_stack(np.where(np.all(img >= 0, axis=2))), axis=1)
    rgb = np.reshape(img, (np.prod(img.shape[:2]), 3))
    pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
    value_base = np.hstack([pixel_numbers, xy_coords, rgb])
    df_base = pd.DataFrame(value_base, columns=['pixel_number', 'X', 'Y', 'R', 'G', 'B'])
    df_base['X'] = df_base['X'] / 240
    df_base['Y'] = df_base['Y'] / 320
    del df_base['pixel_number']
    test_tensor = torch.tensor(df_base[['X', 'Y', 'R', 'G', 'B']].values, dtype=torch.float32).to(device)
    return test_tensor

def post_proc_mlp(model_output: torch.Tensor, image_shape=(320, 240)):
    """ Postprocess model output to get normal map.

    Args: model_output: torch.Tensor of shape (1,3)
    Return: two torch.Tensor of shape (1,3)

    """
    test_np = model_output.reshape(*image_shape, 3)
    normal = copy.deepcopy(test_np)  # surface normal image
    test_np = torch.tensor(test_np,
                           dtype=torch.float32)  # convert to torch tensor for later processing in gradient computation
    test_np = test_np.permute(2, 0, 1)  # swap axes to (3,320,240)
    test_np = test_np # convert to uint8 for visualization
    return test_np, normal

class MLP(nn.Module):
    dropout_p = 0.05

    def __init__(
            self, input_size=5, output_size=3, hidden_size=32):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x

class Digit:

    STREAMS: typing.Dict = {
        # VGA resolution support 30 (default) and 15 fps
        "VGA": {
            "resolution": {"width": 640, "height": 480},
            "fps": {"30fps": 30, "15fps": 15},
        },
        # QVGA resolution support 60 (default) and 30 fps
        "QVGA": {
            "resolution": {"width": 320, "height": 240},
            "fps": {"60fps": 60, "30fps": 30},
        },
    }
    LIGHTING_MIN: int = 0
    LIGHTING_MAX: int = 15
    __LIGHTING_SCALER = 17

    def __init__(self, 
                config: str,
                model_path:str,
                base_image_path:str) -> None:
        """
        DIGIT Device class for a single DIGIT
        :param serial: DIGIT device serial
        :param name: Human friendly identifier name for the device
        """
        self.serial: str = "serial"
        self.name: str = "name"
        self.__dev: typing.Optional[cv2.VideoCapture] = None

        config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        self.sensor_config = config['sensor']

        self.dev_name: str = self.sensor_config['device']
        self.manufacturer: str = ""
        self.model: str = ""
        self.revision: int = ""

        self.resolution: typing.Dict = {}
        self.fps: int = 0
        self.intensity: int = 0

        self.mlp = MLP()
        self.mlp.to(device)
        self.mlp.load_state_dict(torch.load(model_path, map_location=device))

        base_image = cv2.imread(base_image_path)
        self.base_depth = self.infer_depth(base_image)

    def connect(self) -> None:
        logger.info(f"{self.serial}:Connecting to DIGIT")
        self.__dev = cv2.VideoCapture(self.dev_name)
        if not self.__dev.isOpened():
            logger.error(
                f"Cannot open video capture device {self.serial} - {self.dev_name}"
            )
            raise Exception(f"Error opening video stream: {self.dev_name}")
        # set stream defaults, QVGA at 60 fps
        logger.info(
            f"{self.serial}:Setting stream defaults to QVGA, 60fps, maximum LED intensity."
        )
        logger.debug(f"Default stream to QVGA {self.STREAMS['QVGA']['resolution']}")
        self.set_resolution(self.STREAMS["QVGA"])
        logger.debug(f"Default stream with {self.STREAMS['QVGA']['fps']['30fps']} fps")
        self.set_fps(self.STREAMS["QVGA"]["fps"]["30fps"])
        logger.debug("Setting maximum LED illumination intensity")
        # self.set_intensity(15)

    def set_resolution(self, resolution: typing.Dict) -> None:
        """
        Sets stream resolution based on supported streams in Digit.STREAMS
        :param resolution: QVGA or VGA from Digit.STREAMS
        :return: None
        """
        self.resolution = resolution["resolution"]
        width = self.resolution["width"]
        height = self.resolution["height"]
        logger.debug(f"Stream resolution set to {height}w x {width}h")
        self.__dev.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__dev.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_fps(self, fps: int) -> None:
        """
        Sets the stream fps, only valid values from Digit.STREAMS are accepted.
        This should typically be called after the resolution is set as the stream fps defaults to the
        highest fps
        :param fps: Stream FPS
        :return: None
        """
        self.fps = fps
        logger.debug(f"{self.serial}:Stream FPS set to {self.fps}")
        self.__dev.set(cv2.CAP_PROP_FPS, self.fps)

    def set_intensity(self, intensity: int) -> int:
        """
        Sets all LEDs to specific intensity, this is a global control.
        :param intensity: Value between 0 and 15 where 0 is all LEDs off and 15 all
        LEDS full intensity
        :return: Returns the set intensity
        """
        if self.revision < 200:
            # Deprecated version 1.01 (1b) is not supported
            intensity = int(intensity / self.__LIGHTING_SCALER)
            logger.warn(
                "You are using a previous version of the firmware "
                "which does not support independent RGB control, update your DIGIT firmware."
            )
        self.intensity = self.set_intensity_rgb(intensity, intensity, intensity)
        return self.intensity

    def set_intensity_rgb(
        self, intensity_r: int, intensity_g: int, intensity_b: int
    ) -> int:
        """
        Sets LEDs to specific intensity, per LED control
        Perimitted values are between 0 (off/dim) and 15 (full brightness)
        :param intensity_r: Red value
        :param intensity_g: Green value
        :param intensity_b: Blue value
        :return: Returns the set intensity
        """
        if not all(
            [x in range(0, 16) for x in (intensity_r, intensity_g, intensity_b)]
        ):
            raise ValueError("RGB values must be between 0 and 15.")
        intensity = (intensity_r << 8) | (intensity_g << 4) | intensity_b
        logger.debug(
            f"{self.serial}:LED intensity set to {intensity} (R: {intensity_r} G: {intensity_g} B: {intensity_b}"
        )
        self.intensity = intensity
        self.__dev.set(cv2.CAP_PROP_ZOOM, self.intensity)
        return self.intensity

    def get_frame(self, transpose: bool = False) -> np.ndarray:
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """
        ret, frame = self.__dev.read()
        if not ret:
            logger.error(
                f"Cannot retrieve frame data is DIGIT device open?"
            )
            raise Exception(
                f"Unable to grab frame from {self.dev_name}!"
            )
        if not transpose:
            frame = cv2.transpose(frame, frame)
            frame = cv2.flip(frame, 0)
        return frame

    def save_frame(self, path: str) -> np.ndarray:
        """
        Saves a single image frame to host
        :param path: Path and file name where the frame shall be saved to
        :return: None
        """
        frame = self.get_frame()
        logger.debug(f"Saving frame to {path}")
        cv2.imwrite(path, frame)
        return frame

    def get_diff(self, ref_frame: np.ndarray) -> np.ndarray:
        """
        Returns the difference between two frames
        :param ref_frame: Original frame
        :return: Frame difference
        """
        diff = self.get_frame() - ref_frame
        return diff

    def show_view(self, ref_frame: np.ndarray = None) -> None:
        """
        Creates OpenCV named window with live view of DIGIT device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """
        while True:
            frame = self.get_frame()
            if ref_frame is not None:
                frame = self.get_diff(ref_frame)
            cv2.imshow(f"Digit View {self.serial}", frame)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def disconnect(self) -> None:
        logger.info(f"{self.serial}:Closing DIGIT device")
        self.__dev.release()

    def info(self) -> str:
        """
        Returns DIGIT device info
        :return: String representation of DIGIT device
        """
        has_dev = self.__dev is not None
        is_connected = False
        if has_dev:
            is_connected = self.__dev.isOpened()
        info_string = (
            f"Name: {self.dev_name}"
            f"\n\t- Model: {self.model}"
            f"\n\t- Revision: {self.revision}"
            f"\n\t- Connected?: {is_connected}"
        )
        if is_connected:
            info_string += (
                f"\nStream Info:"
                f"\n\t- Resolution: {self.resolution['width']} x {self.resolution['height']}"
                f"\n\t- FPS: {self.fps}"
                f"\n\t- LED Intensity: {self.intensity}"
            )
        return info_string
    
    def __del__(self) -> None:
        if self.__dev is not None and self.__dev.isOpened():
            self.disconnect()

    def infer_depth(self, frame: np.ndarray) -> np.ndarray:
        """ 
            Infer depth from a single image frame

        Args:
            frame: OpenCV image frame

        Returns:
            img_depth: Depth image
        """
        img_np = preproc_mlp(frame)
        img_np = self.mlp(img_np).detach().cpu().numpy()
        img_np, _ = post_proc_mlp(img_np)

        # reconstruct depth
        gradx_img, grady_img = geom_utils._normal_to_grad_depth(img_normal=img_np, gel_width=self.sensor_config['gel_width'],
                                                                gel_height=self.sensor_config['gel_height'], bg_mask=None)
        img_depth = geom_utils._integrate_grad_depth(gradx_img, grady_img, boundary=None, bg_mask=None, max_depth=0.02076)

        return img_depth

    def extract_pcd(self, depth, view_mat = torch.eye(4)):
        """
            Extract point cloud from depth image, subtract base depth image
        
        Args:
            depth: Depth image
            view_mat: View matrix c2w ?
        
        Returns:
            pcd: Point cloud
        """
        proj_mat = torch.tensor(self.sensor_config['P'], dtype=torch.float32)
        
        # Project depth to 3D
        points3d = geom_utils.depth_to_pts3d(depth=depth, P=proj_mat, V=view_mat, params=self.sensor_config)
        points3d = geom_utils.remove_background_pts(points3d, bg_mask=None)
        cloud = o3d.geometry.PointCloud()
        clouds = geom_utils.init_points_to_clouds(clouds=[copy.deepcopy(cloud)], points3d=[points3d])

        return clouds

    def extract_depth(self):
        """ 
            Extract depth from a single image frame 

        Returns:
            frame: OpenCV image frame
            img_depth: Depth image    
        """
        frame = self.get_frame()
        depth = self.infer_depth(frame)

        return frame, depth

def capture_task(opt):
    """
        Capture Camera images and save them to disk
    """
    cam = Digit(
        opt.config,
        opt.model,
        opt.background
    )
    cam.connect()
    os.makedirs(opt.save_dir, exist_ok=True)

    index = 0 
    try:
        while True:
            frame, img_depth = cam.extract_depth()

            cv2.imshow("frame", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            if opt.save_data:
                cv2.imwrite(os.path.join(opt.save_dir, f"frame_{index}.png"), frame)
                index += 1
    
    except KeyboardInterrupt:
        print(" Keyboard interrupt. Exiting... ")
        cam.disconnect()
    
    finally:
        del cam

def point_cloud_task(opt):
    """
        Create point cloud from images
    """
    cam = Digit(
        opt.config,
        opt.model,
        opt.background
    )

    # TODO Point Cloud Inference    
    img = cv2.imread(opt.img)
    depth = cam.infer_depth(img)
    pcd = cam.extract_pcd(depth)
    o3d.visualization.draw_geometries(pcd)
    
    return 

def rerun_depth(opt):
    """
        Create point cloud from images
    """
    cam = Digit(
        opt.config,
        opt.model,
        opt.background
    )

    # TODO Point Cloud Inference    
    fns = sorted(glob(f"{opt.data_dir}/touch_frames/*.png"))
    for fn in tqdm(fns):
        img = cv2.imread(fn)
        depth = cam.infer_depth(img)
        depth_u16 = (depth.numpy() * 1e6).astype(np.uint16)
        cv2.imwrite(fn.replace("/touch_frames/", "/touch_depths/"), depth_u16)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="../../config/digit.yaml", help="Digit Config")
    args.add_argument("--model", type=str, default="/home/user/Documents/ActiveTouch/src/gaussian_splatting/weights/digit.pth", help="Path to model weights")
    args.add_argument("--background", type=str, default="/home/user/Documents/ActiveTouch/src/gaussian_splatting/weights/background.png", help="Background image for digit sensor")
    args.add_argument("--save_data", action="store_true", default=False, help="Save images to disk")
    args.add_argument("--save_dir", type=str, default="/home/user/Documents/digit-depth/images", help="Directory to save images")
    args.add_argument("--img", type=str, default="/home/user/Documents/frame_0.png", help="Input image path")
    args.add_argument("--task", type=str, default="capture", choices=["capture", "point_cloud", "rerun"], help="Task to perform")
    args.add_argument("--data_dir", type=str, default="")
    opt = args.parse_args()

    if opt.task == "capture":
        capture_task(opt)
    elif opt.task == "point_cloud":
        point_cloud_task(opt)
    elif opt.task == "rerun":
        rerun_depth(opt)

    

    
