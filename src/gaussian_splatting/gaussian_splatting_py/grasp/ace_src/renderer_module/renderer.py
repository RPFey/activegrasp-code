import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..misc import EasyDict
from .rays import *


def sample_pdf(bins, weights, n_pts_per_ray, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_pts_per_ray, 1. - 0.5 / n_pts_per_ray, steps=n_pts_per_ray)
        u = u.expand(list(cdf.shape[:-1]) + [n_pts_per_ray])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_pts_per_ray])

    # Invert CDF
    u = u.to(cdf.device).contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self, sdf_network, decoder_input,
                 device,
                 deviation_network,
                 n_rays, n_pts_per_ray,
                 n_importance, up_sample_steps, sample_range) -> None:
        # 传递 SDF 的 decoder 函数
        if decoder_input == 'pos':
            self.decode_sdf = lambda pos, features: sdf_network(pos, features, input=pos)
        elif decoder_input == 'None':
            self.decode_sdf = lambda pos, features: sdf_network(pos, features)
        else:
            raise NotImplementedError
        self.decode_gradient = sdf_network.gradient

        self.device = device
        self.n_rays = n_rays
        self.n_pts_per_ray = n_pts_per_ray
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.sample_range = sample_range
        self.deviation_network = deviation_network

    def get_regular_rays(self, intrinsics, extrinsics, new_resolution=None):
        '''根据相机内参、外参, 为每个像素点计算一条光线 (世界坐标系下 ray_origins + r * ray_dirs)
        Args:
            intrinsic: (B, 6), [fx, fy, cx, cy, width, height]
            extrinsic: (B, 4, 4)
            new_resolution: [width, height] 指定分辨率的话进行降采样, 同时渲染整张图像
        Returns:
            ray_origins: (B, RAY_NUM, 3)
            ray_dirs: (B, RAY_NUM, 2)
            注意: RAY 的排序从像素坐标系的左上角开始, 从左到右, 从上到下 [H, W].reshape(-1)
        '''
        ray_origins, ray_dirs = [], []

        B = extrinsics.shape[0]

        # breakpoint()
        T_world2cam = extrinsics.cpu()
        T_cam2world = torch.stack([torch.inverse(T_world2cam[n]) for n in range(B)], dim=0).to(self.device)

        # 外参转4*4矩阵
        cam_locs_world = T_cam2world[:, :3, 3]  # 从 w2c 转换到 c2w, 提取平移向量
        fx = intrinsics[:, 0]
        fy = intrinsics[:, 1]
        cx = intrinsics[:, 2]
        cy = intrinsics[:, 3]
        resolution = int(intrinsics[0, 4]), int(intrinsics[0, 5])
        if new_resolution[0] != resolution[0] or new_resolution[1] != resolution[1]:
            fx = fx * new_resolution[0] / resolution[0]
            fy = fy * new_resolution[1] / resolution[1]
            cx = cx * new_resolution[0] / resolution[0]
            cy = cy * new_resolution[1] / resolution[1]
            resolution = new_resolution

        # 构造 [0-w, 0-h] 的均匀网格, 摊开到 (RAY_NUM, 2) [像素坐标系]
        u_axis = torch.arange(resolution[0], dtype=torch.float32, device=intrinsics.device) + 0.5
        v_axis = torch.arange(resolution[1], dtype=torch.float32, device=intrinsics.device) + 0.5
        uv = torch.stack(torch.meshgrid(u_axis, v_axis))
        uv = uv.transpose(1, 2)  # 交换行列排布
        uv = uv.reshape(2, -1).transpose(1, 0)  # 2, resolution, resolution -> resolution**2, 2
        # 重复 B 次, (B, RAY_NUM, 2)
        uv = uv.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)

        # 构造 x_cam, y_cam 网格对[像素坐标] (B, RAY_NUM)
        x_cam = uv[:, :, 0].view(B, -1)
        y_cam = uv[:, :, 1].view(B, -1)

        # 添加一维 z_cam[相机坐标/mm]
        z_lift = torch.ones(x_cam.shape, device=intrinsics.device)

        # 从[像素坐标]转换到[相机坐标/mm]
        x_lift = (x_cam - cx.unsqueeze(-1)) / fx.unsqueeze(-1) * z_lift  # 注意广播
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_lift

        # 得到齐次坐标点 (x,y,z=1,1), [相机坐标/mm], (B, RAY_NUM, 4)
        cam_rel_points = torch.stack((x_lift, y_lift, z_lift, torch.ones_like(z_lift)), dim=-1)
        # 和相机外参相乘, 得到世界坐标系下的坐标点, [世界坐标/mm], (B, RAY_NUM, 3)
        world_rel_points = torch.bmm(T_cam2world, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]

        # 得到光线单位方向
        ray_dirs = world_rel_points - cam_locs_world[:, None, :]  # None 添加维度, 注意广播
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)  # 归一化, 注意这里会变成斜线距离而不是 z 距离了

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs

    def gen_random_ray_index(self, batch_size, resolution, num_rays):
        # return torch.randint(low=0, high=resolution, size=[batch_size, num_rays])  # 存在重复
        index = [torch.randperm(resolution)[:num_rays] for i in range(batch_size)]
        return torch.stack(index, dim=0)

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """Up sampling give a fixed inv_s; 根据 sdf 以及 inv_s 计算权重, 用于下一次重要性采样
        """
        batch_size, n_rays, n_pts_per_ray, _ = z_vals.shape
        pts = (rays_o.unsqueeze(-2) + z_vals * rays_d.unsqueeze(-2)).reshape(batch_size, n_rays, n_pts_per_ray, 3)  # n_rays, n_pts_per_ray, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)  # 最大奇异值? 注意不是二范数 batch_size, n_rays, n_pts_per_ray
        inside_sphere = (radius[..., :-1] < 1.0) | (radius[..., 1:] < 1.0)
        sdf = sdf.reshape(batch_size, -1, n_pts_per_ray)
        prev_sdf, next_sdf = sdf[..., :-1], sdf[..., 1:]
        prev_z_vals, next_z_vals = z_vals[..., :-1, 0], z_vals[..., 1:, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5  # 从 section 点转到 mid-point
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # 光线方向和 SDF 梯度的夹角

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros_like(cos_val[..., :1]), cos_val[..., :-1]], dim=-1)  # cos 往后移一位
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)  # 到此为止相当于是宽度为 2 的滑动窗口取 min
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere  # 去掉正值(和 SDF 同向), 去掉球外部

        dist = (next_z_vals - prev_z_vals)  # 采样点之间的距离
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # 从 mid-point 转到前后两个点
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)  # 大φ
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)  # 公式 13
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, n_rays, 1]).to(self.device), 1. - alpha + 1e-7], -1), -1)[..., :-1]

        z_vals = z_vals.reshape(batch_size * n_rays, -1)
        weights = weights.reshape(batch_size * n_rays, -1)
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()  # 再采样 n_importance 个点
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, feature, last=False):
        '''把新采样的 z 值加入到原来的 z 值中, 并排序; 更新对应 sdf
        '''
        batch_size, n_rays, n_pts_per_ray, _ = z_vals.shape
        new_z_vals = new_z_vals.reshape(batch_size, n_rays, -1, 1)
        _, _, n_importance, _ = new_z_vals.shape  # eg [4, 512, 16, 1]
        # pts = (rays_o.unsqueeze(-2) + new_z_vals * rays_d.unsqueeze(-2)).reshape(batch_size*n_rays, -1, 3)  # eg [4, 8192, 3]

        z_vals = torch.cat([z_vals, new_z_vals], dim=-2)  # eg [4, 512, 64+16, 1]
        z_vals, index = torch.sort(z_vals, dim=-2)  # eg [4, 512, 80, 1]  对 z 排序

        if not last:  # 计算 new_z point 的 sdf, 根据 index 合并
            pts = (rays_o.unsqueeze(-2) + z_vals * rays_d.unsqueeze(-2)).reshape(batch_size, -1, 3)  # eg [4, 512 * 80, 3]
            sdf = self.decode_sdf(pts, feature).unsqueeze(-1)  # eg [4, 512 * 80, 1]

        return z_vals, sdf

    def get_rays(self, W, H, n_rays, decoder_padding,
                 original_size, bounding_box,
                 camera_intrinsic_matrix,
                 camera_extrinsic_matrix,
                 use_selected_rays):
        batch_size = camera_extrinsic_matrix.shape[0]
        # 生成光线起点和方向
        raw_ray_origins, raw_ray_dirs = self.get_regular_rays(camera_intrinsic_matrix,
                                                              camera_extrinsic_matrix,  # 注意这里只选一张侧视图做监督
                                                              new_resolution=(W, H))
        # 采样一部分光线
        if use_selected_rays:
            ray_index = self.gen_random_ray_index(batch_size, int(W * H), n_rays).to(self.device)  # 随机采样光线
        else:
            ray_index = torch.arange(0, n_rays, 1, device=self.device).unsqueeze(0).repeat(batch_size, 1)  # 返回全图
        # 这里不能用 index_select 因为对每个 scene 都采样了不同的光线
        ray_origins = torch.gather(raw_ray_origins, 1, ray_index.unsqueeze(-1).repeat(1, 1, 3))
        ray_dirs = torch.gather(raw_ray_dirs, 1, ray_index.unsqueeze(-1).repeat(1, 1, 3))

        return ray_origins, ray_dirs, ray_index

    def get_points(self, ray_origins, ray_dirs, ray_index, W, H, feature,
                   original_size, decoder_padding, n_pts_per_ray, n_importance, bounding_box,
                   sample_depth_near_input=True, iteration_count=None, depth_img=None):
        # 每条光线上可采样的深度范围
        # 判断光线是否与 bounding box 相交
        near, far, valid_ray_mask = get_rays_intersection(ray_origins, ray_dirs, bounding_box)  # [batch_size, n_rays, 1] * 2
        # valid_ray_mask = (near > 0.0) & (far > 0.0)  # 不相交的光线, [batch_size, n_rays, 1]
        if sample_depth_near_input:
            # 在 sensor input depth 附近采点
            near_, far_ = get_rays_range_from_gt(depth_img[:, 0, :],
                                                 W, H, ray_index,
                                                 offset_range=self.get_sample_depth_near_input_range(iteration_count))
            cross_box_before_table = far <= 0.5 * (near_ + far_)  # 排除侧穿 box 的光线, 此处也包括了不相交的情况
            valid_ray_mask[cross_box_before_table] = False
            # 用更精细的 near_ 和 far_ 替换 near 和 far
            replace_flag = near_ > near
            near = near * ~replace_flag + near_ * replace_flag
            replace_flag = far_ < far
            far = far * ~replace_flag + far_ * replace_flag
            # old version:
            # 判断 near_ 和 far_ 是否在 near 和 far 之间
            # 如果成立, 则用 near_ 和 far_ 替换 near 和 far
            # replace_flag = (near_ > near) & (far_ < far)
            # near = near * ~replace_flag + near_ * replace_flag
            # far = far * ~replace_flag + far_ * replace_flag

        near = torch.clamp(near, min=0.0)  # 考虑到 padding 之后相机在空间内部的情况

        # 每条光线上采样点的深度(叠加一点随机量)
        z_vals = torch.linspace(0.0, 1.0, n_pts_per_ray).to(self.device)
        z_vals = near + (far - near) * z_vals  # [batch_size, n_rays, n_pts_per_ray], 注意广播
        batch_size = z_vals.shape[0]
        n_rays = z_vals.shape[1]
        t_rand = (torch.rand([batch_size, n_rays, n_pts_per_ray]) - 0.5).to(self.device)
        z_vals = z_vals + valid_ray_mask * t_rand * 2.0 / n_pts_per_ray
        z_vals = z_vals.unsqueeze(-1)  # -> [batch_size, n_rays, n_pts_per_ray, 1]

        # importance sample
        if n_importance > 0:
            points = (ray_origins.unsqueeze(-2) + z_vals * ray_dirs.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sdf = self.decode_sdf(points, feature).unsqueeze(-1)
            for i in range(self.up_sample_steps):
                new_z_vals = self.up_sample(ray_origins,
                                            ray_dirs,
                                            z_vals,
                                            sdf,
                                            self.n_importance // self.up_sample_steps,
                                            64 * 2**i)
                z_vals, sdf = self.cat_z_vals(ray_origins,
                                              ray_dirs,
                                              z_vals,
                                              new_z_vals,
                                              sdf,
                                              feature,
                                              last=(i + 1 == self.up_sample_steps))  # 最后一轮就不用计算 sdf 了
        n_pts_per_ray = n_pts_per_ray + n_importance
        # 至此，得到了一个 z_vals(添加了扰动的且进行了重要性采样)

        sample_dist = 2.0 / n_pts_per_ray

        # Section length
        dists = z_vals[..., 1:, :] - z_vals[..., :-1, :]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).to(self.device).expand(dists[..., :1, :].shape)], -2)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints 拿到光线上 mid point 的方向向量、距离
        pts = ((ray_origins.unsqueeze(-2) + mid_z_vals * ray_dirs.unsqueeze(-2))
               * valid_ray_mask.unsqueeze(-1)).reshape(batch_size, -1, 3)  # batch * 光线 * 深度 * 3 空间坐标

        # limit the points to the bounding box
        pts = torch.clamp(pts, bounding_box[0], bounding_box[3])
        # real world coordinates -> [-0.5, 0.5]
        pts -= original_size * 0.5
        pts = pts / original_size / (1 + decoder_padding)
        dirs = ray_dirs.unsqueeze(-2).repeat(1, 1, n_pts_per_ray, 1)
        dirs = dirs.reshape(batch_size, -1, 3)
        return pts, dirs, mid_z_vals, dists, valid_ray_mask

    def get_cos_anneal_ratio(self, iteration_count):
        '''一开始是 0, 逐渐增加到 1
        '''
        iteration_end = 20000.0
        if iteration_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, iteration_count / iteration_end])

    def get_sample_depth_near_input_range(self, iteration_count):
        '''采样点深度的范围, 从 input 的 depth 出发, 前后延伸一定范围
        '''
        range_start = 0.05
        range_end = 0.4
        iteration_end = 20000
        if iteration_count > iteration_end:
            return range_end
        else:
            return range_start + (range_end - range_start) * iteration_count / iteration_end

    def render(self,
               feature,
               camera_intrinsic_matrix,
               camera_extrinsic_matrix,
               depth_img,
               bounding_box, original_size, decoder_padding,
               iteration_count,
               **kwargs):
        batch_size = depth_img.shape[0]
        render_out = EasyDict()

        # default parameters
        n_pts_per_ray = self.n_pts_per_ray
        W, H = camera_intrinsic_matrix[0, 4:6]
        n_rays = self.n_rays
        use_selected_rays = True
        sample_depth_near_input = True
        render_normal = False
        n_importance = self.n_importance
        # rewrite parameters
        if 'use_selected_rays' in kwargs:
            use_selected_rays = kwargs['use_selected_rays']
        if 'new_n_pts_per_ray' in kwargs:
            n_pts_per_ray = kwargs['new_n_pts_per_ray']
        if 'new_resolution' in kwargs:  # [width, height] 指定分辨率的话进行降采样, 同时渲染整张图像
            W, H = kwargs['new_resolution']
            n_rays = int(W * H)
        if 'sample_depth_near_input' in kwargs:
            sample_depth_near_input = kwargs['sample_depth_near_input']
        if 'render_normal' in kwargs:
            render_normal = kwargs['render_normal']
        if 'n_importance' in kwargs:
            n_importance = kwargs['n_importance']

        # get rays
        ray_origins, ray_dirs, ray_index = self.get_rays(W, H, n_rays, decoder_padding,
                                                         original_size, bounding_box,
                                                         camera_intrinsic_matrix,
                                                         camera_extrinsic_matrix,
                                                         use_selected_rays)

        # get points
        pts, dirs, mid_z_vals, dists, valid_ray_mask = self.get_points(ray_origins, ray_dirs, ray_index, W, H, feature,
                                                                       original_size, decoder_padding, n_pts_per_ray, n_importance, bounding_box,
                                                                       sample_depth_near_input=sample_depth_near_input,
                                                                       iteration_count=iteration_count,
                                                                       depth_img=depth_img)
        # get sdf from feature plane
        sdf = self.decode_sdf(pts, feature).unsqueeze(-1)  # [batch_size, points, 1]

        # compute depth
        depth, weights, inv_s, cos_anneal_ratio, gradients, gradient_error = self.compute_depth(
            batch_size, pts, mid_z_vals, feature, dirs, sdf, dists, n_rays, iteration_count)

        # compute normal
        if render_normal:
            normals = self.compute_normal(batch_size, gradients, weights,
                                          n_rays, camera_extrinsic_matrix)
            render_out.normal_imgs = normals

        # results
        render_out.points = pts
        render_out.sdf = sdf
        render_out.mid_z_vals = mid_z_vals
        render_out.rendered_depth = depth
        render_out.ray_index = ray_index
        render_out.valid_ray_mask = valid_ray_mask
        render_out.gradient_error = gradient_error
        # statistics
        render_out.s_val = (1.0 / inv_s).mean()
        render_out.cos_anneal_ratio = cos_anneal_ratio
        render_out.sample_depth_near_input_range = self.get_sample_depth_near_input_range(iteration_count)
        return render_out

    def compute_depth(self, batch_size, pts, mid_z_vals, feature, dirs, sdf, dists, n_rays, iteration_count):
        '''从 SDF 等信息合成一条 ray 上的 depth
        '''
        n_pts_per_ray = mid_z_vals.shape[-2]  # after importance sampling
        inv_s = self.deviation_network(torch.zeros([1, 3]).to(self.device))[:, :1].clip(1e-6, 1e6)  # single parameter net
        inv_s = inv_s.expand(batch_size, n_pts_per_ray * n_rays, 1)

        gradients = self.decode_gradient(pts, feature).squeeze()  # 计算输出 SDF 对 点坐标 的梯度
        true_cos = (dirs * gradients).sum(-1, keepdim=True)  # 求 光线方向 和 SDF 梯度方向 的夹角余弦值

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        cos_anneal_ratio = self.get_cos_anneal_ratio(iteration_count)  # 一个训练时候的 trick 总之就是对 cos 施加一个非线性激活层
        iter_cos = - (F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                   + F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points 从 mid point 转到 节点
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(batch_size, -1, 1) * 0.5
        estimated_next_sdf = sdf + iter_cos * dists.reshape(batch_size, -1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf  # 微分变差分 体现在这里
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size * n_rays, n_pts_per_ray).clip(0.0, 1.0)  # 对应公式 13 即权重, shape [512, 128]

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size * n_rays, n_pts_per_ray)  # 求一个范数, 然后判断是否在球内
        inside_sphere = (pts_norm < 1.0).float().detach()  # 一般是一头一尾容易超出球体
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size * n_rays, 1]).to(self.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        depth = (mid_z_vals.reshape(-1, n_pts_per_ray, 1) * weights[:, :, None]).sum(dim=1)
        depth = depth.reshape(batch_size, n_rays)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size * n_rays,
                                                              n_pts_per_ray,  # ↓ 注意这里默认三平面
                                                              -1), ord=2, dim=-1) - 1.0) ** 2
                                                              # self.cfg.encoder.c_dim * 3), ord=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        return depth, weights, inv_s, cos_anneal_ratio, gradients, gradient_error

    def compute_normal(self, batch_size, gradients, weights, n_rays, camera_extrinsic_matrix):
        n_pts_per_ray = weights.shape[-1]
        normals = gradients.reshape(batch_size * n_rays, n_pts_per_ray, -1) * weights[:, :, None]  # SDF 梯度 * 权重
        normals = normals.sum(dim=1).reshape(batch_size, n_rays, -1)
        rot = camera_extrinsic_matrix[:, :3, :3]  # 获取相机旋转矩阵
        for i in range(batch_size):
            normals[i] = torch.matmul(rot[[i]], normals[i, ..., None]).squeeze(-1)  # 世界坐标系转相机坐标系
        return normals

    def render_full_image(self,
                          feature,
                          camera_intrinsic_matrix,
                          camera_extrinsic_matrix,
                          depth_img,
                          bounding_box, original_size, decoder_padding,
                          iteration_count,
                          **kwargs):
        '''针对 完整图像 渲染的简化版 render 过程 (分 batch), very similar to self.render
        '''
        batch_size = depth_img.shape[0]
        render_out = EasyDict()

        # default parameters
        n_pts_per_ray = self.n_pts_per_ray
        W, H = camera_intrinsic_matrix[0, 4:6]
        n_rays = self.n_rays
        use_selected_rays = False
        sample_depth_near_input = True
        n_importance = self.n_importance
        # rewrite parameters
        if 'new_n_pts_per_ray' in kwargs:
            n_pts_per_ray = kwargs['new_n_pts_per_ray']
        if 'new_resolution' in kwargs:  # [width, height] 指定分辨率的话进行降采样, 同时渲染整张图像
            W, H = kwargs['new_resolution']
            n_rays = int(W * H)
        if 'sample_depth_near_input' in kwargs:
            sample_depth_near_input = kwargs['sample_depth_near_input']
        if 'n_importance' in kwargs:
            n_importance = kwargs['n_importance']

        n_rays_per_batch = 1024
        if n_rays % n_rays_per_batch != 0:
            raise ValueError('n_rays % n_rays_per_batch != 0')
        batch_count = batch_size * n_rays // n_rays_per_batch  # batch_count * n_rays_per_batch = batch_size * n_rays

        # get rays
        raw_ray_origins, raw_ray_dirs, raw_ray_index = self.get_rays(W, H, n_rays, decoder_padding,
                                                                     original_size, bounding_box,
                                                                     camera_intrinsic_matrix,
                                                                     camera_extrinsic_matrix,
                                                                     use_selected_rays)

        # split rays [batch_size, n_rays] -> [batch_count, n_rays_per_batch]
        ray_origins = raw_ray_origins.reshape(batch_count, n_rays_per_batch, 3)
        ray_dirs = raw_ray_dirs.reshape(batch_count, n_rays_per_batch, 3)
        ray_index = raw_ray_index.reshape(batch_count, n_rays_per_batch)

        depth_all = torch.zeros(batch_count, n_rays_per_batch, device=self.device)
        normals_all = torch.zeros(batch_count, n_rays_per_batch, 3, device=self.device)
        valid_ray_mask_all = torch.zeros(batch_count, n_rays_per_batch, 1, dtype=torch.bool, device=self.device)

        # for i in tqdm(range(batch_count)):
        for i in range(batch_count):
            reference_index = i // (batch_count // batch_size)  # after scale up, rearrange depth and feature
            reference_depth = depth_img[[reference_index]]
            reference_feature = {}
            for key in feature.keys():
                reference_feature[key] = feature[key][[reference_index]]
            # get points
            pts, dirs, mid_z_vals, dists, valid_ray_mask = self.get_points(ray_origins[[i]], ray_dirs[[i]], ray_index[[i]], W, H, reference_feature,
                                                                           original_size, decoder_padding, n_pts_per_ray, n_importance, bounding_box,
                                                                           sample_depth_near_input=sample_depth_near_input,
                                                                           iteration_count=iteration_count,
                                                                           depth_img=reference_depth)
            # get sdf from feature plane
            sdf = self.decode_sdf(pts, reference_feature).unsqueeze(-1)  # [batch_size, points, 1]

            # compute depth
            depth, weights, inv_s, cos_anneal_ratio, gradients, gradient_error = self.compute_depth(
                1, pts, mid_z_vals, reference_feature, dirs, sdf, dists, n_rays_per_batch, iteration_count)

            # compute normal
            normals = self.compute_normal(1, gradients, weights,
                                          n_rays_per_batch, camera_extrinsic_matrix)
            # write this batch
            depth_all[i] = depth.detach()
            normals_all[i] = normals.detach()
            valid_ray_mask_all[i] = valid_ray_mask.detach()

        render_out.rendered_depth = depth_all
        render_out.normal_imgs = normals_all
        render_out.valid_ray_mask = valid_ray_mask_all
        return render_out

    def get_orthographic_rays(self, W, H, n_rays, decoder_padding,
                              original_size, bounding_box, batch_size=1):
        '''产生正交的光线(非针孔相机) 仅用于渲染 mesh 或 SDF 截面
        '''
        # 生成网格的起点 和竖直向下的方向
        raw_ray_dirs = torch.tensor([0, 0, -1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_rays, 1)

        u_axis = torch.arange(W, dtype=torch.float32, device=self.device) / W + 0.5 / W
        u_axis = (u_axis - 0.5) * original_size * (1 + decoder_padding) + original_size * 0.5
        v_axis = torch.arange(H, dtype=torch.float32, device=self.device) / H + 0.5 / H
        v_axis = (v_axis - 0.5) * original_size * (1 + decoder_padding) + original_size * 0.5
        uv = torch.stack(torch.meshgrid(u_axis, v_axis))
        uv = uv.transpose(1, 2)  # 交换行列排布
        uv = uv.reshape(2, -1).transpose(1, 0)  # 2, resolution, resolution -> resolution**2, 2
        uv = uv.unsqueeze(0).repeat(batch_size, 1, 1)
        z = torch.zeros_like(uv[:, :, 0]) + bounding_box[5] + 0.1
        raw_ray_origins = torch.stack([uv[:, :, 0], uv[:, :, 1], z], dim=-1)  # B, n_rays, 3
        return raw_ray_origins, raw_ray_dirs

    def extract_geometry(self,
                         feature,
                         resolution,
                         bounding_box, original_size, decoder_padding,
                         iteration_count,
                         batch_size=1,
                         **kwargs):
        '''针对 mesh 渲染的简化版 render 过程 (分 batch)
        '''
        assert list(feature.values())[0].shape[0] == 1, 'only extract geometry from single scene'
        render_out = EasyDict()
        # default parameters
        n_rays_per_batch = 1024
        batch_count = resolution**2 // n_rays_per_batch  # batch_count * n_rays_per_batch = resolution**2
        if resolution**2 % n_rays_per_batch != 0:
            raise ValueError('resolution**2 % n_rays_per_batch != 0')

        n_importance = 0
        # rewrite parameters
        W = H = n_pts_per_ray = resolution
        n_rays = int(W * H)

        # get rays
        raw_ray_origins, raw_ray_dirs = self.get_orthographic_rays(W, H, n_rays, decoder_padding,
                                                                   original_size, bounding_box,
                                                                   batch_size=batch_size)
        # split rays
        ray_origins = raw_ray_origins.reshape(batch_count, n_rays_per_batch, 3)  # [1, resolution**2, 3] -> [batch_count, n_rays_per_batch, 3]
        ray_dirs = raw_ray_dirs.reshape(batch_count, n_rays_per_batch, 3)
        ray_index = None
        sdf = torch.zeros(batch_count, n_rays_per_batch * resolution, 1, device=self.device)

        # for i in tqdm(range(batch_count)):
        for i in range(batch_count):
            pts, dirs, mid_z_vals, dists, valid_ray_mask = self.get_points(ray_origins[[i]], ray_dirs[[i]], ray_index, W, H, feature,
                                                                           original_size, decoder_padding,
                                                                           n_pts_per_ray, n_importance, bounding_box,
                                                                           sample_depth_near_input=False,
                                                                           iteration_count=iteration_count)
            # pts: [1, n_rays_per_batch * resolution, 3]
            # sdf: [1, n_rays_per_batch * resolution, 1]
            sdf[[i]] = self.decode_sdf(pts, feature).unsqueeze(-1).detach()

        render_out.sdf = sdf.reshape(1, -1, 1)
        return render_out
