import cv2
import open3d as o3d
import math
import numpy  as np
from scipy.spatial.transform import Rotation as SciR
from gaussian_splatting_py.digit.run_digit import Digit

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Canvas
from tqdm import tqdm

def convert_T(trans, rot):
    T = np.eye(4)
    T[:3, :3] = SciR.from_quat(rot).as_matrix()
    T[:3, 3] = trans
    return T

# realsense frame
camera_T = convert_T(np.array([0.346, -0.078, 0.394]), np.array([0.714, 0.681, 0.120, 0.106]))
# left inner pad
# left_inner_finger_pad_T = convert_T(np.array([0.357, -0.034, 0.028]), np.array([-0.638, 0.770, 0.004, 0.005]))
# right inner pad
right_inner_finger_pad_T = convert_T(np.array([0.406, 0.038, 0.052]), np.array([0.053, 0.997, 0.006, -0.052]))

ARUCO_DICT = cv2.aruco.DICT_5X5_250
MARKER_SIZE = 0.10 # meter
BOARDER_WIDTH = 0.016 # meter

rs_intrinsic = np.array([
    [381.2928466796875, 0, 310.50665283203125],
    [0, 380.92254638671875, 245.01397705078125],
    [0, 0, 1]
])

objps = np.array([
    (BOARDER_WIDTH + MARKER_SIZE, BOARDER_WIDTH , 0), 
    (BOARDER_WIDTH, BOARDER_WIDTH, 0),
    (BOARDER_WIDTH, BOARDER_WIDTH + MARKER_SIZE, 0),
    (BOARDER_WIDTH + MARKER_SIZE, BOARDER_WIDTH + MARKER_SIZE, 0)
])

def get_marker_corner(ids=0):
    col_idx = ids // 6
    if col_idx % 2 == 1:
        row_idx = 2 * (ids % 6)
    else:
        row_idx =  2 * (ids % 6) + 1

    base = np.array([[col_idx * 0.030 + 0.0035, row_idx * 0.030 + 0.0035, 0.0]])

    corner = np.array([
        [0, 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [1., 0., 0.]   
    ]) * 0.023

    return base + corner

# lower left, lower right, upper point
# triangle_points = np.array([
#     [0.23, 0.072, 0.01],
#     [0.23, 0.078, 0.01],
#     [0.23 - 0.003 * math.sqrt(3), 0.075, 0.01]
# ])

# For the vertical triangle
triangle_points = (np.array([[188., 75., 10.]]) + np.array([
    [50., 0., 15.],
    [50., 0., 25.],
    [50. - 5 * np.sqrt(3), 0., 20.]
])) / 1000.

# # Outer Facing Board
# [0.606, -0.007, 0.002], 
# [0.580, -0.159, 0.001],
# [0.434, -0.127, -0.002],
# [0.458, 0.017, -0.002]

# Inner Facing Board 
# [0.636, 0.061, 0.003]
# [0.619, -0.086, 0.001] 
# [0.474, -0.060, -0.003]
# [0.489, 0.081, -0.002]

def calculate_board_pose():
    points = np.array([
        [0.636, 0.061, 0.001],
        [0.619, -0.086, 0.001], 
        [0.474, -0.060, 0.001],
        [0.489, 0.081, 0.001],
    ])

    board_points = np.array([
        [0, 0, 0],
        [0, 0.15, 0],
        [0.15, 0.15, 0],
        [0.15, 0, 0]
    ])

    # procrustes analysis
    centroid_points = np.mean(points, axis=0)
    centroid_board = np.mean(board_points, axis=0)

    points_centralized = points - centroid_points
    board_points_centralied = board_points - centroid_board

    H = np.dot(board_points_centralied.T, points_centralized)
    U, S, Vt = np.linalg.svd(H)
    S = np.diag([1, 1, np.linalg.det(np.dot(Vt.T, U.T))])
    R = Vt.T @ S @ U.T
    t = centroid_points - np.dot(R, centroid_board)

    board_pose = np.eye(4)
    board_pose[:3, :3] = R
    board_pose[:3, 3] = t

    return board_pose

def cam2digit(args):
    # # detect aruco markers
    # img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)    

    # aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    # aruco_params = cv2.aruco.DetectorParameters()
    # aruco_params.markerBorderBits = 1

    # corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    # # ret, rvec, tvec = cv2.solvePnP(objps, corners[0], rs_intrinsic, np.zeros(5), flags = cv2.SOLVEPNP_IPPE)
    # # draw markers
    # detection = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    # cv2.imshow("img", detection)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # obj_coords = [get_marker_corner(i) for i in ids.reshape(-1)]
    # model_points = np.concatenate(obj_coords, axis=0)
    # cam_points = np.concatenate(corners, axis=0)
    # cam_points = cam_points.reshape(-1, 2)
    # rvet, rvec, tvec = cv2.solvePnP(model_points, cam_points, rs_intrinsic, np.zeros(5), flags=cv2.SOLVEPNP_IPPE) #, rvec=init_rvecs, tvec=init_tvecs, useExtrinsicGuess=True)
    # b2c = np.eye(4)
    # b2c[:3, :3] = cv2.Rodrigues(rvec)[0]
    # b2c[:3, 3] = tvec.reshape(-1)
    # print("Board to World: ", b2c)
    # # board in world
    # b2w = np.dot(camera_T, b2c)

    b2w = calculate_board_pose()
    print("Board to word: ", b2w)   
    digit_frame = cv2.imread(args.touch)

    digit = Digit(args.config, args.model, args.bg)
    projmat = np.array(digit.sensor_config['P'])

    # get camera pose
    fl_y = projmat[1, 1]
    fl_x = projmat[0, 0]
    cx = digit_frame.shape[1] / 2
    cy = digit_frame.shape[0] / 2
    digit_intrinsic = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])

    # user interface for clicking, the interior points
    points_in_image = start_app(args.touch)
    points_in_image = np.array(points_in_image, dtype=np.float32)
    
    # solve P3P
    rvet, rvec, tvec = cv2.solveP3P(triangle_points, points_in_image, digit_intrinsic, np.zeros(5), cv2.SOLVEPNP_P3P)
    rvec = rvec[0]
    tvec = tvec[0]

    board_2_digit = np.eye(4)
    board_2_digit[:3, :3] = cv2.Rodrigues(rvec)[0]
    board_2_digit[:3, 3] = tvec.reshape(-1)

    digit_in_board = np.linalg.inv(board_2_digit)
    digit_in_word = np.dot(b2w, digit_in_board)

    digit_in_ee = np.linalg.inv(right_inner_finger_pad_T) @ digit_in_word
    
    postion = digit_in_ee[:3, 3]
    print("Postion: ", postion)

    rot = SciR.from_matrix(digit_in_ee[:3, :3]).as_quat()
    print("Rotation: ", rot)
    
def start_app(image_path):
    # Points and labels storage
    points = []
    img = cv2.imread(image_path)

    # bind mouse event 
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            # remove the last point
            if len(points) > 0:
                points.pop()

            print(points)
        
        # draw the points
        circle = img.copy()
        for point in points:
            circle = cv2.circle(circle, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv2.imshow("Digit", circle)


    cv2.namedWindow("Digit", cv2.WINDOW_NORMAL)
    cv2.imshow("Digit", img)
    cv2.setMouseCallback("Digit", on_click)
    cv2.waitKey(0)

    return np.array(points)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--img", type=str, required=True, help="Path to Camera Image")
    args.add_argument("--touch", type=str, default="../touch.png", help="Path to Touch Image")
    args.add_argument("--config", type=str, default="../config/digit.yaml", help="Path to Digit Config")
    args.add_argument("--model", type=str, default="../weights/digit.pth", help="Path to Digit Model")
    args.add_argument("--bg", type=str, default="../weights/background.png", help="Path to Digit Background Image")

    opt = args.parse_args()
    cam2digit(opt)

