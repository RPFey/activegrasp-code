import torch
import cv2


def _get_intersection_of_ray_and_box(ray_origins, ray_dirs, bounding_box):
    ''' Get the intersection of a ray with a bounding box, return near and far.
        【works fine with small data, but not support tensor】
    Args:
        ray_origins, ray_dirs: [B, N_rays, 3]
        bounding_box: [x_min, y_min, z_min, x_max, y_max, z_max]
    Return:
        near, far: [B, N_rays, 1]
        if the ray does not hit the box, return 0, 0

    Example:
    >>> ray_origins = torch.tensor([[[ 0.0718, -0.1989, -0.9774],
                                     [ 0.3273, -0.5609, -0.7604],
                                     [ 0.3181, -0.1485, -0.9364]]])
    >>> ray_dirs = torch.tensor([[[ 0.1966,  0.2907,  0.5658],
                                  [ 0.1966,  0.2907,  0.5658],
                                  [ 0.1966,  0.2907,  0.5658]]])
    >>> bounding_box = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]
    >>> near, far = get_intersection_of_ray_and_box(ray_origins, ray_dirs, bounding_box)

    Codes from: https://blog.csdn.net/u012325397/article/details/50807880
    '''
    batch_size, N_rays = ray_origins.shape[:2]

    near, far = torch.zeros([batch_size, N_rays, 1], device=ray_origins.device), torch.zeros([batch_size, N_rays, 1], device=ray_origins.device)

    def _ray_hit_box(origin, dir, bounding_box):
        ox, oy, oz = origin
        dx, dy, dz = dir
        x0, y0, z0, x1, y1, z1 = bounding_box

        # check if ray is parallel to the box
        if dx == 0 and (ox < x0 or ox > x1):
            return 0, 0
        if dy == 0 and (oy < y0 or oy > y1):
            return 0, 0
        if dz == 0 and (oz < z0 or oz > z1):
            return 0, 0

        # calculate the intersection of the ray with the surfaces of the box
        if dx > 0:
            tx_min = (x0 - ox) / dx
            tx_max = (x1 - ox) / dx
        else:
            tx_min = (x1 - ox) / dx
            tx_max = (x0 - ox) / dx
        if dy > 0:
            ty_min = (y0 - oy) / dy
            ty_max = (y1 - oy) / dy
        else:
            ty_min = (y1 - oy) / dy
            ty_max = (y0 - oy) / dy
        if dz > 0:
            tz_min = (z0 - oz) / dz
            tz_max = (z1 - oz) / dz
        else:
            tz_min = (z1 - oz) / dz
            tz_max = (z0 - oz) / dz
        t0 = max(tx_min, ty_min, tz_min, 0)
        t1 = min(tx_max, ty_max, tz_max)

        # check if the ray intersects the box
        if t0 > t1:
            return 0, 0
        else:
            return t0, t1

    for i in range(batch_size):
        for j in range(N_rays):
            near[i, j], far[i, j] = _ray_hit_box(ray_origins[i, j], ray_dirs[i, j], bounding_box)

    return near, far


def get_rays_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, bounding_box):
    """
    Args:
        bounding_box: [x_min, y_min, z_min, x_max, y_max, z_max]

    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    # Returns (-1, -1) for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)

    # bb_min = [-1*(box_side_length/2), -1*(box_side_length/2), -1*(box_side_length/2)]
    # bb_max = [1*(box_side_length/2), 1*(box_side_length/2), 1*(box_side_length/2)]
    # bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)

    bounds = torch.tensor(bounding_box, dtype=rays_o.dtype, device=rays_o.device).reshape(2, 3)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.  通过方向向量的符号来选择碰撞的面(在 bounds 的第一个维度中)
    # 例如这里用 x 的符号来选择碰撞 x_min 或是 x_max
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.  记录不相交的情况
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.  is_valid 记录每个 ray 的情况, 在这里用作 bool 型的高级索引
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -1

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1), is_valid.reshape(*o_shape[:-1], 1)


def get_rays_range_from_gt(depth_img, W, H, ray_index, offset_range=0.01):
    """
    Args:
        depth_img: [B, H_origin, W_origin]
        W, H: int
        ray_index: [B, N_rays]
    Returns:
        near: [B, N_rays, 1]
        far: [B, N_rays, 1]
    """
    near = torch.zeros(depth_img.shape[0], ray_index.shape[1], 1, device=depth_img.device)
    far = torch.zeros(depth_img.shape[0], ray_index.shape[1], 1, device=depth_img.device)

    offset_range /= 2
    depth_img = depth_img.cpu().numpy()
    W = int(W)
    H = int(H)

    # 缩放, 然后用 ray_index 索引 sample 的光线的真实深度
    for index, img in enumerate(depth_img):
        new_img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        new_img = new_img.reshape(-1)
        near[index] = torch.tensor(new_img)[ray_index[index]].unsqueeze(-1) - offset_range
        far[index] = torch.tensor(new_img)[ray_index[index]].unsqueeze(-1) + offset_range
    return near, far


if __name__ == '__main__':
    ray_origins = torch.tensor([[[ 0.1966,  0.2907,  0.5658]]])
    # ray_origins = torch.tensor([[[ 0.1966,  0.2507,  0.2658]]])
    ray_dirs = torch.tensor([[[ 0.0718, -0.1989, -0.9774],
                              [ 0.3273, -0.5609, -0.7604],
                              [ 0.3181, -0.1485, -0.9364],
                              [-0.2978, -0.3553, -0.8861],
                              [-0.1392, -0.0480, -0.9891],
                              [-0.0614, -0.5772, -0.8143],
                              [-0.3440, -0.1375, -0.9289],
                              [-0.2156, -0.4551, -0.8639],
                              [-0.1350,  0.0946, -0.9863],
                              [ 0.0390, -0.3498, -0.9360],
                              [-0.2078, -0.3578, -0.9104],
                              [ 0.2391,  0.1617, -0.9574],
                              [ 0.0624,  0.2473, -0.9669],
                              [ 0.1267, -0.3884, -0.9128],
                              [ 0.2769, -0.1851, -0.9429],
                              [ 0.0425,  0.3265, -0.9442],
                              [ 0.2132, -0.2439, -0.9461],
                              [ 0.1409, -0.1678, -0.9757],
                              [ 0.4843,  0.0156, -0.8748],
                              [ 0.4039, -0.0699, -0.9121]]])
    ray_origins = ray_origins.repeat(1, ray_dirs.shape[1], 1)
    bounding_box = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]
    
    # near, far = _get_intersection_of_ray_and_box(ray_origins, ray_dirs, bounding_box)

    near, far = get_rays_range(ray_origins, ray_dirs, bounding_box)
