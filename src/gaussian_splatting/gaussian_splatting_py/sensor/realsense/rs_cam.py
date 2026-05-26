
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as sciR
from multiprocessing import Manager
import multiprocessing as mp
import numpy as np
import cv2
import time

class RSCamera:
    def __init__(self) -> None:
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            raise Exception("The demo requires Depth camera with Color sensor")
        
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)

        # device = profile.get_device()
        # depth_sensor = device.first_depth_sensor()
        # device.hardware_reset()

        # depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , depth_scale)

        # clipping_distance_in_meters = 1 
        # clipping_distance = clipping_distance_in_meters / depth_scale

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_intrinsics(self):
        """
        Get the intrinsics of the camera

        Returns:
            depth_intrinsics: rs.intrinsics
            color_intrinsics: rs.intrinsics
        """
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()

        return depth_intrinsics, color_intrinsics

    def capture(self):
        """
        Capture a depth and color image from the camera

        Returns:
            depth_image: np.array (H, W) np.uint16
            color_image: np.array (H, W, 3) np.uint8
        """

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Convert images to numpy arrays
        # copy them so the frames source could be releases
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        color_image = np.asanyarray(color_frame.get_data()).copy()

        return depth_image, color_image
    
    def __del__(self):
        print("Closing camera pipeline")
        self.pipeline.stop()

    @property
    def color_fx(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.fx

    @property
    def color_fy(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.fy

    @property
    def color_cx(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.ppx

    @property
    def color_cy(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.ppy
    
    @property
    def color_height(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.height
    
    @property
    def color_width(self):
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        return color_intrinsics.width
    
    def intrinsic_matrix(self):
        """
        Get the intrinsic matrix of the camera
        """
        intrinsics = self.get_intrinsics()
        color_intrinsics = intrinsics[1]
        intrinsic_matrix = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ]
        )
        return intrinsic_matrix
    
    def rel_transform(self):
        """ 
            Relative transformation matrix that transforms points from realsense to the built-in camera
        """
        rel_rot = self.rel_rotation()
        rel_tran = self.rel_translation()
        rel_transform = np.eye(4)
        rel_transform[:3, :3] = sciR.from_quat(rel_rot, scalar_first=False).as_matrix()
        rel_transform[:3, 3] = rel_tran
        return rel_transform

class RealSenseCamera:
    def __init__(self):
        # Create a queue to store the frames, limited to 10 frames
        self.frame_queue = mp.Manager().list()
        # Create a lock for synchronized access to the queue
        self.lock = mp.Manager().Lock()
        self.kill_event = mp.Manager().Event()
        self.kill_event.clear()
        self._get_intrinsics()
        
        # Create a process to handle camera capture
        self.camera_process = mp.Process(target=self._capture_frames, args=(self.frame_queue, self.lock, self.kill_event))
        # Start the process
        self.camera_process.start()

    def _get_intrinsics(self):
        pipeline = rs.pipeline()
        config = rs.config()

        fps = 30 
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

        # Start streaming
        profile = pipeline.start(config)
        profile = pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        # stop camera
        pipeline.stop()
        self.color_intrinsics = color_intrinsics

    def _capture_frames(self, frame_list, lock, kill_event):
        # Initialize the Intel RealSense pipeline
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        fps = 30 
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)

        # Start streaming
        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        try:
            while True:
                if kill_event.is_set():
                    break

                # Wait for a frame from the camera
                try:
                    frames = pipeline.wait_for_frames()
                except RuntimeError as e:
                    print(" Fail to capture frame ... ")
                    pipeline.stop()
                    time.sleep(2)

                    print("Restarting camera stream")
                    pipeline.start(config)
                    continue

                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                # copy them so the frames source could be releases
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                color_image = np.asanyarray(color_frame.get_data()).copy()

                # Safely append the frame to the queue using a lock
                with lock:
                    if len(frame_list) == 0:
                        frame_list.append([color_image, depth_image])
                    else:
                        frame_list[0] = [color_image, depth_image]

                # Add a slight delay for stability
                time.sleep(1 / fps)

        except KeyboardInterrupt:
            pass

        # Stop the camera stream when done
        print("Stopping camera stream")
        pipeline.stop()

    def capture(self):
        if not self.camera_process.is_alive():
            print("Camera process is not running, restart the process")
            self.camera_process.start()

        try:
            while True:
                with self.lock:
                    if len(self.frame_queue) > 0:
                        # Return the end frame in the queue
                        color, depth = self.frame_queue[0]
                        color, depth = color.copy(), depth.copy()
                        self.frame_queue.pop()
                        return depth, color
                    else:
                        print("No frame in the queue, sleep 1 s")
                        time.sleep(1)
        
        except KeyboardInterrupt:
            exit()

    def stop(self):
        # Stop the camera process
        self.kill_event.set()
        self.camera_process.join()
        self.kill_event.clear()

    def __del__(self):
        self.stop()

    @property
    def color_fx(self):
        return self.color_intrinsics.fx

    @property
    def color_fy(self):
        return self.color_intrinsics.fy

    @property
    def color_cx(self):
        return self.color_intrinsics.ppx

    @property
    def color_cy(self):
        return self.color_intrinsics.ppy
    
    @property
    def color_height(self):
        return self.color_intrinsics.height
    
    @property
    def color_width(self):
        return self.color_intrinsics.width

    def get_intrinsics(self):
        return self.color_intrinsics
    
    def intrinsic_matrix(self):
        """
        Get the intrinsic matrix of the camera
        """
        intrinsic_matrix = np.array(
            [
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1]
            ]
        )
        return intrinsic_matrix

if __name__ == "__main__":
    import argparse
    import os
    import json

    cam = RealSenseCamera()
    args = argparse.ArgumentParser()
    args.add_argument("--save_dir", default="/", type=str)
    args.add_argument("--save_data", action="store_true", default=False)
    opt = args.parse_args()

    index = 0
    if opt.save_data:
        intrinsics = cam.get_intrinsics()
        print(intrinsics)

        os.makedirs(opt.save_dir, exist_ok=True)
        os.makedirs(os.path.join(opt.save_dir, "color"), exist_ok=True)
        os.makedirs(os.path.join(opt.save_dir, "depth"), exist_ok=True)
    
    try:
        while True:
            depth_image, color_image = cam.capture()
            if depth_image is None or color_image is None:
                continue
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(100)

            if opt.save_data:
                # write images
                cv2.imwrite(os.path.join(opt.save_dir, "color", "{:04d}.png".format(index)), color_image)
                cv2.imwrite(os.path.join(opt.save_dir, "depth", "{:04d}.png".format(index)), depth_image)

                index += 1
    
    except KeyboardInterrupt:
        if opt.save_data:
            cam_intrinsics = np.array(
                [
                    [intrinsics[0].fx, 0, intrinsics[0].ppx],
                    [0, intrinsics[0].fy, intrinsics[0].ppy],
                    [0, 0, 1]
                ]
            )
            cam_intrinsics = [v.tolist() for v in cam_intrinsics]

            rgbd_intrinsics = np.array(
                [
                    [intrinsics[1].fx, 0, intrinsics[1].ppx],
                    [0, intrinsics[1].fy, intrinsics[1].ppy],
                    [0, 0, 1]
                ]
            )
            rgbd_intrinsics = [v.tolist() for v in rgbd_intrinsics]
            stats = {
                "color_intrinsics": cam_intrinsics,
                "depth_intrinsics": rgbd_intrinsics
            }

            # save the intrinsics
            with open(os.path.join(opt.save_dir, "color_intrinsics.json"), "w") as f:
                json.dump(stats, f)

    finally:
        cv2.destroyAllWindows()
        del cam

        
            