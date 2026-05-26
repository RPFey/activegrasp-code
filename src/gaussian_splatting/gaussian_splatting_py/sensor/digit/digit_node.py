#!/usr/bin/env python3

import numpy as np
import cv2
import json
from typing import List, Union
import matplotlib
from dataclasses import dataclass
from collections import namedtuple
import matplotlib as mpl

import tf2_ros as tf2
from scipy.spatial.transform import Rotation as sciR
# from kortex_driver.srv import DoSensorFocusActionRequest, DoSensorFocusAction

import threading
import rospy
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest

from gaussian_splatting_py.digit.run_digit import Digit

class DigitNode(object):

    def __init__(self) -> None:
        rospy.init_node("digit_node")

        # fetch parameter
        self.model_path = rospy.get_param("~model_path", "")
        self.config_path = rospy.get_param("~config_path", "")
        self.base_img_path = rospy.get_param("~base_img_path", "")
        rospy.loginfo("Model path: {}".format(self.model_path))
        rospy.loginfo("Config path: {}".format(self.config_path))
        rospy.loginfo("Base image path: {}".format(self.base_img_path))

        try:
            self.digit = Digit(self.config_path, self.model_path, self.base_img_path)
            self.digit.connect()
        except Exception as e:
            rospy.logerr("Failed to connect to Digit: {}".format(e))
            # del self.digit
            exit()

        self.digit_color_pub = rospy.Publisher("~digit_color", Image, queue_size=10)
        self.digit_depth_pub = rospy.Publisher("~digit_depth", Image, queue_size=10)
        self.digit_depth_viz = rospy.Publisher("~digit_depth_viz", Image, queue_size=10)

        # set timer 
        self.timer = rospy.Timer(rospy.Duration(0.1), self.digit_pub_thread)

    def digit_pub_thread(self, event) -> None:
        """ Publish digit prediction to ROS topic """
        
        frame = self.digit.get_frame()
        depth = self.digit.infer_depth(frame)
        
        # convert to numpy array
        depth = depth.numpy()
        depth_u16 = (depth * 1e6).astype(np.uint16)

        # colorize
        cmap = mpl.cm.viridis
        depth_vis = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth_vis = cmap(depth_vis)[:, :, :3] * 255
        depth_vis = depth_vis.astype(np.uint8)

        color = Image()
        color.header.stamp = rospy.Time.now()
        color.header.frame_id = "digit_color"
        color.height = frame.shape[0]
        color.width = frame.shape[1]
        color.encoding = "rgb8"
        color.data = frame.tobytes()

        depth_img = Image()
        depth_img.header.stamp = rospy.Time.now()
        depth_img.header.frame_id = "digit_depth"
        depth_img.height = depth_u16.shape[0]
        depth_img.width = depth_u16.shape[1]
        depth_img.encoding = "mono16"
        depth_img.data = depth_u16.tobytes()

        depth_vis_img = Image()
        depth_vis_img.header.stamp = rospy.Time.now()
        depth_vis_img.header.frame_id = "digit_depth_vis"
        depth_vis_img.height = depth_vis.shape[0]
        depth_vis_img.width = depth_vis.shape[1]
        depth_vis_img.encoding = "rgb8"
        depth_vis_img.data = depth_vis.tobytes()

        self.digit_color_pub.publish(color)
        self.digit_depth_pub.publish(depth_img)
        self.digit_depth_viz.publish(depth_vis_img)


if __name__ == "__main__":
    node = DigitNode()
    rospy.spin()