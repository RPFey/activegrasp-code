from ..vgn_src.perception import *


def pixel2cam(u, v, depth, intrinsic) -> np.ndarray:
    '''
    input: 像素位置和深度 (u, v, depth)
    output: 相机坐标系下的坐标 (x, y, z), 注意 z=depth
    '''
    intrinsic = _get_intrinsic_matrix(intrinsic)
    pos = np.linalg.inv(intrinsic) @ np.array([u, v, 1]) * depth
    return pos


def cam2pixel(x, y, z, intrinsic) -> np.ndarray:
    '''
    input: 相机坐标系下的坐标 (x, y, z)
    output: 像素位置 (u, v)
    '''
    intrinsic = _get_intrinsic_matrix(intrinsic)
    pos = intrinsic @ np.array([x, y, z])
    pos = pos / pos[2]
    return pos[:2].astype(np.int32)


def cam2world(x, y, z, extrinsic) -> np.ndarray:
    '''
    input: 相机坐标系下的坐标 (x, y, z)
    output: 世界坐标系下的坐标 (U, V, W)
    '''
    extrinsic = _get_extrinsic_matrix(extrinsic)
    pos = np.array([x, y, z, 1])
    pos = np.linalg.inv(extrinsic) @ pos
    return pos[:3]


def world2cam(U, V, W, extrinsic) -> np.ndarray:
    '''
    input: 世界坐标系下的坐标 (U, V, W)
    output: 相机坐标系下的坐标 (x, y, z)
    '''
    extrinsic = _get_extrinsic_matrix(extrinsic)
    pos = np.array([U, V, W, 1])
    pos = extrinsic @ pos
    return pos[:3]


def pixel2world(u, v, depth, intrinsic, extrinsic) -> np.ndarray:
    '''
    input: 像素位置和深度 (u, v, depth)
    output: 世界坐标系下的坐标 (U, V, W)
    '''
    pos = pixel2cam(u, v, depth, intrinsic)
    pos = cam2world(*pos, extrinsic)
    return pos


def world2pixel(U, V, W, intrinsic, extrinsic) -> np.ndarray:
    '''
    input: 世界坐标系下的坐标 (U, V, W)
    output: 像素位置 (u, v, depth)
    '''
    pos = world2cam(U, V, W, extrinsic)
    pos = cam2pixel(*pos, intrinsic)
    return pos


def _get_intrinsic_matrix(intrinsic):
    if isinstance(intrinsic, CameraIntrinsic):
        return intrinsic.K
    elif isinstance(intrinsic, np.ndarray) and intrinsic.shape == (3, 3):
        return intrinsic
    else:
        raise ValueError("intrinsic error")


def _get_extrinsic_matrix(extrinsic):
    '''注意 外参是 w2c 矩阵
    '''
    if isinstance(extrinsic, Transform):
        return extrinsic.as_matrix()
    elif isinstance(extrinsic, np.ndarray) and extrinsic.shape == (4, 4):
        return extrinsic
    else:
        raise ValueError("extrinsic error")
