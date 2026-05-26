import numpy as np
import cv2

from torchvision import transforms

def convert_intrinsics(img, old_intrinsics = (360.01, 360.01, 243.87, 137.92), new_intrinsics = (1297.67, 1298.63, 620.91, 238.28), new_size=(1280, 720)):
    """
    Convert a set of images to a different set of camera intrinsics.
    Parameters:
    - images: List of input images.
    - old_intrinsics: Tuple (fx, fy, cx, cy) of the old camera intrinsics.
    - new_intrinsics: Tuple (fx, fy, cx, cy) of the new camera intrinsics.
    - new_size: Tuple (width, height) defining the size of the output images.
    Returns:
    - List of images converted to the new camera intrinsics.
    """
    old_fx, old_fy, old_cx, old_cy = old_intrinsics
    new_fx, new_fy, new_cx, new_cy = new_intrinsics
    width, height = new_size
    
    # Constructing the old and new intrinsics matrices
    K_old = np.array([[old_fx, 0, old_cx], [0, old_fy, old_cy], [0, 0, 1]])
    K_new = np.array([[new_fx, 0, new_cx], [0, new_fy, new_cy], [0, 0, 1]])
    # Compute the inverse of the new intrinsics matrix for remapping
    K_new_inv = np.linalg.inv(K_new)
    
    # Construct a grid of points representing the new image coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    homogenous_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=-1).T
    
    # Convert to the old image coordinates
    old_coords = K_old @ K_new_inv @ homogenous_coords
    old_coords /= old_coords[2, :]  # Normalize to make homogeneous
    
    # Reshape for remapping
    map_x = old_coords[0, :].reshape(height, width).astype(np.float32)
    map_y = old_coords[1, :].reshape(height, width).astype(np.float32)
    
    # Remap the image to the new intrinsics
    converted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return converted_img


def warp_image(image, K, R, t):
    """
    Warp an image from the perspective of camera 1 to camera 2.

    :param image: Input image from camera 1
    :param K: Intrinsic matrix of both cameras
    :param R: Rotation matrix from camera 1 to camera 2
    :param t: Translation vector from camera 1 to camera 2
    :return: Warped image as seen from camera 2
    """
    # Compute the homography matrix
    H = compute_homography(K, R, t)

    # Warp the image using the homography
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image


def compute_homography(K, R, t):
    """
    Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K_inv = np.linalg.inv(K)
    H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
    return H


def resize_if_too_large(image):
    if image.shape[0] > 1600 and image.shape[1] > 1600:
        image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
    return image


def preprocess_image(image):
    # Transform to convert image to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def learn_scale_and_offset_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = x
    return scale, offset

CHECHERBOARD_SIZE = (8, 11)
CHECHERBOARD_SIZE_IN_METER = 0.03

# checkerboard size
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

def compute_pose_from_marker(img, mtx, dist):
    """ Compute the pose of the camera from the marker 
    
    Args:
    - img: Image of the marker
    - mtx: Camera matrix
    - dist: Distortion coefficients
    - viz: Visualize the detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.markerBorderBits = 1
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    detection = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imshow('img', detection)
    cv2.waitKey(500)

    obj_coords = [get_marker_corner(i) for i in ids.reshape(-1)]
    model_points = np.concatenate(obj_coords, axis=0)
    
    cam_points = np.concatenate(corners, axis=0)
    cam_points = cam_points.reshape(-1, 2)
    rvet, rvec, tvec = cv2.solvePnP(model_points, cam_points, mtx, dist, flags=cv2.SOLVEPNP_IPPE) #, rvec=init_rvecs, tvec=init_tvecs, useExtrinsicGuess=True)

    return rvet, rvec, tvec

def calib_cam(img_paths, use_default=False):
    """ Calibrate camera from checker board images 
    
    Returns:
        rvecs: rotation vectors (w2c)
        tvecs: translation vectors (w2c)
    """

    rvecs = []
    tvecs = [] 

    if use_default:
        # IF you want to use existing camera matrix and distortion coefficients
        dist = np.zeros((1, 5))
        if 'rs' in img_paths[0]:
            # realsense camera
            mtx = np.array([
                [381.2928466796875, 0, 310.50665283203125],
                [0, 380.92254638671875, 245.01397705078125],
                [0, 0, 1]
            ])
        else:
            # kinova
            mtx = np.array([
                [1297.672904, 0, 620.914026],
                [0, 1298.631344, 238.280325],
                [0, 0, 1]
            ])
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        num_obj_points = CHECHERBOARD_SIZE[0] * CHECHERBOARD_SIZE[1]
        objp = np.zeros((num_obj_points, 3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECHERBOARD_SIZE[0], 0:CHECHERBOARD_SIZE[1]].T.reshape(-1,2) * CHECHERBOARD_SIZE_IN_METER
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # compute intrinsics
        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECHERBOARD_SIZE, None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECHERBOARD_SIZE, corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(500)
        
        # cv2.destroyAllWindows()
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Camera Matrix: \n", mtx)
        print("Distortion Coefficients: \n", dist)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        rvet, rvec, tvec = compute_pose_from_marker(img, mtx, dist)
        
        rvec = rvec.reshape(-1)
        tvec = tvec.reshape(-1)

        rvecs.append(rvec)
        tvecs.append(tvec)

    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return rvecs, tvecs

def calib_cam(img_paths, use_default=False):
    """ Calibrate camera from checker board images 
    
    Returns:
        rvecs: rotation vectors (w2c)
        tvecs: translation vectors (w2c)
    """

    rvecs = []
    tvecs = [] 

    if use_default:
        # IF you want to use existing camera matrix and distortion coefficients
        dist = np.zeros((1, 5))
        if 'rs' in img_paths[0]:
            # realsense camera
            mtx = np.array([
                [381.2928466796875, 0, 310.50665283203125],
                [0, 380.92254638671875, 245.01397705078125],
                [0, 0, 1]
            ])
        else:
            # kinova
            mtx = np.array([
                [1297.672904, 0, 620.914026],
                [0, 1298.631344, 238.280325],
                [0, 0, 1]
            ])
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        num_obj_points = CHECHERBOARD_SIZE[0] * CHECHERBOARD_SIZE[1]
        objp = np.zeros((num_obj_points, 3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECHERBOARD_SIZE[0], 0:CHECHERBOARD_SIZE[1]].T.reshape(-1,2) * CHECHERBOARD_SIZE_IN_METER
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # compute intrinsics
        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECHERBOARD_SIZE, None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECHERBOARD_SIZE, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        
        # cv2.destroyAllWindows()
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Camera Matrix: \n", mtx)
        print("Distortion Coefficients: \n", dist)

    for img_path in img_paths:
        img = cv2.imread(img_path)
        rvet, rvec, tvec = compute_pose_from_marker(img, mtx, dist)
        
        rvec = rvec.reshape(-1)
        tvec = tvec.reshape(-1)

        rvecs.append(rvec)
        tvecs.append(tvec)

    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)

    return rvecs, tvecs