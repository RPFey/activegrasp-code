import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize

from .decoder_module import decoder
from .encoder_module import voxels
from .renderer_module.single_variance import SingleVarianceNetwork
from .renderer_module.rays import *
from .renderer_module.renderer import *
from .misc import EasyDict

from ..vgn_src.transform import Rotation, Transform

def normal2RM(normal: np.ndarray) -> Rotation:
    '''法向 → 旋转矩阵 (固定一个额外的 x 轴)
    注意 数据生成、后续抓取验证 yaw 转四元数 要遵循同样的写法
    Args:
        normal: [3] 法向
    '''
    # 根据抓取法向建立坐标轴 (给定 z, 随便来个正交 xy)  define initial grasp frame on object surface
    z_axis = -normal  # z 轴：点云法向的反向（朝物体内）
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):  # 判断 xz 内积是否接近 1（差值小于 1e-4）
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)  # 叉乘
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)  # 从法向向量生成旋转矩阵
    return R

class ACENet(nn.Module):
    ''' Network definition. Modified from GIGA.

    Args:
        decoder_type:
        encoder_type:
        c_dim: dimension of latent code c
        padding:
        detach_tsdf: for GIGADetach
        device: torch device
    '''
    def __init__(self, cfg):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.cfg = cfg

        # construct encoder
        dataset = None
        c_dim = cfg.encoder.c_dim
        padding = cfg.padding
        plane_resolution = cfg.encoder.plane_resolution

        encoder_type = self.cfg.encoder.type
        encoder_kwargs = {'plane_type': ['xz', 'xy', 'yz'],
                          'plane_resolution': plane_resolution, 'unet': True,
                          'unet_kwargs':
                          {'depth': 3, 'merge_mode': 'concat', 'start_filts': 32}}
        if encoder_type == 'idx':
            self.encoder = nn.Embedding(len(dataset), c_dim).to(self.device)
        elif encoder_type is not None:  # True
            self.encoder = voxels.LocalVoxelEncoder(
                c_dim=c_dim, padding=padding,
                **encoder_kwargs
            ).to(self.device)
        else:
            self.encoder = None

        # construct decoders
        self.supervision = cfg.geometry_decoder.type
        self.decoder_padding = cfg.decoder_padding
        self.detach_tsdf = False

        decoder_kwargs = {'sample_mode': self.cfg.grasp_decoder.sample_mode,
                          'hidden_size': self.cfg.grasp_decoder.hidden_size,
                          'concat_feat': self.cfg.grasp_decoder.concat_feat}

        grasp_expand_sample_pts = np.prod(cfg.grasp_decoder.expand_sample_pts[:3])
        grasp_sample_along_ray = cfg.grasp_decoder.expand_sample_pts[4] > 0

        if self.cfg.grasp_decoder.decoder_input in ['pos', 'dir']:
            grasp_in_dim = 3
        elif self.cfg.grasp_decoder.decoder_input == 'pos_dir':
            grasp_in_dim = 6
        else:
            grasp_in_dim = 0
        if cfg.geometry_decoder.decoder_input == 'pos':
            geo_in_dim = 3
        else:
            geo_in_dim = 0

        self.decoder_qual = decoder.LocalDecoder(
            in_dim=grasp_in_dim, c_dim=c_dim, padding=padding, out_dim=1,  # input_dim 3 for xyz coordinate
            expand_sample_pts=grasp_expand_sample_pts, with_ray_feature=grasp_sample_along_ray, for_grasp=True,
            **decoder_kwargs).to(self.device)
        self.decoder_rot = decoder.LocalDecoder(
            in_dim=grasp_in_dim, c_dim=c_dim, padding=padding, out_dim=1,
            expand_sample_pts=grasp_expand_sample_pts, with_ray_feature=grasp_sample_along_ray, for_grasp=True,
            **decoder_kwargs).to(self.device)
        self.decoder_width = decoder.LocalDecoder(
            in_dim=grasp_in_dim, c_dim=c_dim, padding=padding, out_dim=1,
            expand_sample_pts=grasp_expand_sample_pts, with_ray_feature=grasp_sample_along_ray, for_grasp=True,
            **decoder_kwargs).to(self.device)
        self.decoder_geo = decoder.LocalDecoder(
            in_dim=geo_in_dim, c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs).to(self.device)

        if self.supervision == "rendered_depth":
            self.deviation_network = SingleVarianceNetwork(init_val=0.3).to(self.device)
            self.renderer = NeuSRenderer(self.decoder_geo, cfg.geometry_decoder.decoder_input,
                                         self.device,
                                         self.deviation_network,
                                         cfg.geometry_decoder.n_rays,
                                         cfg.geometry_decoder.n_pts_per_ray,
                                         cfg.geometry_decoder.n_importance,
                                         cfg.geometry_decoder.up_sample_steps,
                                         cfg.geometry_decoder.sample_range)

    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def forward(self, batch_dict, iteration_count=None,
                grasp_branch=True, geometry_branch=True,
                **kwargs):
        ''' Performs a forward pass through the network.
        Returns: (dict)
            grasp:
                grasp_label: B*1, grasp quality
                grasp_rotation: B*1*4, grasp rotation (四元数)
                grasp_width: B*1, grasp width
            geometry supervision: (↓ or)
                rendered_depth: B*H*W
                occupancy: B*2048*1, occupancy values (density+variance)
        '''
        pred_dict = EasyDict()

        # encode tsdf
        assert len(batch_dict.tsdf.shape) == 4, "tsdf should be 4D (B, R**3)"
        feature = self.encoder(batch_dict.tsdf)
        if batch_dict.tsdf.shape[0] == 1:  # 针对 同一场景 多个目标点
            for key in feature.keys():
                feature[key] = feature[key].repeat(batch_dict.point_grasp.shape[0], 1, 1, 1)

        # e.g. feature [batch_size=32, c_dim=32, 40, 40]

        # decode grasp
        if grasp_branch:
            pred_dict.grasp_label, pred_dict.grasp_rotation, pred_dict.grasp_width = \
                self.decode_grasp(batch_dict, feature)

        # decode geometry
        if geometry_branch:
            if self.supervision == "rendered_depth":
                original_size = 0.3
                bounding_box = self.get_bounding_box(self.decoder_padding, original_size)

                camera_extrinsic_matrix = torch.tensor([Transform.from_list(list(batch_dict.camera_extrinsic[:, 0, :][n].cpu())).as_matrix()
                                                        for n in range(batch_dict.camera_extrinsic.shape[0])],
                                                       device=self.device)

                # render
                render_out = self.renderer.render(feature,
                                                  batch_dict.camera_intrinsic,
                                                  camera_extrinsic_matrix,
                                                  batch_dict.depth_img,
                                                  bounding_box, original_size, self.decoder_padding,
                                                  iteration_count,
                                                  **kwargs)
                for k, v in render_out.items():
                    pred_dict[k] = v

                # approximate sdf
                img_gt = batch_dict.depth_img[:, 0, :]
                img_gt = img_gt.reshape(img_gt.shape[0], -1)
                pixel_gt = torch.stack([img_gt[b, pred_dict.ray_index[b, :]] for b in range(img_gt.shape[0])], dim=0)
                pred_dict.approximate_sdf = pixel_gt[:, :, None, None] - pred_dict.mid_z_vals

            elif self.supervision == "occupancy":
                points = batch_dict.point_occ
                pred_dict.occupancy = torch.sigmoid(self.decoder_geo(points, feature,
                                                                     input=points if self.cfg.geometry_decoder.decoder_input == 'pos' else None
                                                                     )[:, :, 0])

            elif self.supervision == "none":
                pass

            else:
                raise ValueError(f"Unknown supervision type: {self.supervision}")

        return pred_dict

    def get_bounding_box(self, padding, original_size):
        '''generate sample space after padding (real world coordinates)
        '''
        original_size = 0.3
        bounding_box = np.array([0.0, 0.0, 0.0, original_size, original_size, original_size])  # [x_min, y_min, z_min, x_max, y_max, z_max]
        bounding_box -= original_size * 0.5
        bounding_box *= (1 + padding)
        bounding_box += original_size * 0.5
        return bounding_box

    def decode_grasp(self, input_batch, feature, **kwargs):
        ''' Returns grasp prediction for the sampled points.
        Args:
            pos (tensor): points (-0.5 ~ 0.5), [batch_size, 1, 3]
            feature (tensor): latent conditioned code, dict['xz', 'xy', 'yz' or 'grid']
        '''
        batch_size = len(input_batch.point_grasp)
        pos = input_batch.point_grasp
        pos = pos / (1 + self.decoder_padding)

        # 旋转矩阵转单位向量
        direction_vector = torch.zeros_like(pos, device=self.device)
        for i in range(batch_size):
            camera_rotation = Rotation.from_quat(list(input_batch.camera_extrinsic[:, 0, :][i].cpu())[:4])
            camera_M = Transform(camera_rotation, np.array([0, 0, 1])).as_matrix()  # 旋转到光轴方向 然后沿着 z 前进 1 (不用归一化了)
            vector = np.linalg.inv(camera_M)[:3, 3]  # 前进之后在世界坐标下的位置
            direction_vector[i, 0] = torch.from_numpy(vector)

        if self.cfg.grasp_decoder.decoder_input == 'pos':
            input = pos.clone()
        elif self.cfg.grasp_decoder.decoder_input == 'dir':
            input = direction_vector
        elif self.cfg.grasp_decoder.decoder_input == 'pos_dir':
            input = torch.cat([pos.clone(), direction_vector], dim=-1)  # cat 空间位置和方向
        elif self.cfg.grasp_decoder.decoder_input == 'None':
            input = None
        else:
            raise NotImplementedError

        pos = self.expand_sample_pts(pos, direction_vector)  # 扩充采样点 -> [Batch_size, N_pts, 3]

        qual = self.decoder_qual(pos, feature, input, **kwargs)
        qual = torch.where(torch.isnan(qual), torch.zeros_like(qual), qual)  # prevent nan
        qual = torch.sigmoid(qual)
        qual = qual.squeeze(-1)
        rot = self.decoder_rot(pos, feature, input, **kwargs)
        rot = rot.squeeze(1) * np.pi  # 映射到 [0, pi] 之间
        width = self.decoder_width(pos, feature, input, **kwargs)
        width = width.squeeze(-1)
        # squeeze, [B, 1], [B, 1], [B, 1] -> [B], [B], [B]
        return qual, rot, width

    def expand_sample_pts(self, pts: torch.Tensor, dir: torch.Tensor) -> torch.Tensor:
        '''
        pts: [B, 1, 3]
        dir: [B, 1, 3]
        output: [B, 1 + expanded_num, 3]
        '''
        x_num, y_num, z_num, step, along_ray_step = self.cfg.grasp_decoder.expand_sample_pts
        result = []
        for i, p in enumerate(pts):
            # 先找到这个平面上两个正交的向量
            local_x = torch.cross(dir[i][0], torch.tensor([0., 0., 1.], device=self.device, dtype=dir[i][0].dtype))
            local_x = F.normalize(local_x, dim=-1)
            local_y = torch.cross(dir[i][0], local_x)
            local_y = F.normalize(local_y, dim=-1)
            local_z = dir[i][0]

            expanded_pts = [p[0]]
            if np.prod(self.cfg.grasp_decoder.expand_sample_pts[:3]) > 1:
                # 规划格点
                for z_index in range(z_num):
                    for y_index in range(y_num):
                        for x_index in range(x_num):
                            expanded_pts.append(
                                p[0] +
                                step * (x_index - (x_num-1) / 2) * local_x +
                                step * (y_index - (y_num-1) / 2) * local_y +
                                step * (z_index - (z_num-1) / 2) * local_z
                            )
            if self.cfg.grasp_decoder.expand_sample_pts[4] > 0:
                # 加上路径点
                pt = p[0].clone()
                while True:
                    pt_in_box = (pt[0] > -0.5 and pt[0] < 0.5 and
                                 pt[1] > -0.5 and pt[1] < 0.5 and
                                 pt[2] > -0.5 and pt[2] < 0.5)
                    if pt_in_box:
                        expanded_pts.append(pt.clone())
                    else:
                        break
                    pt += along_ray_step * local_z  # 第一个点从零开始, 否则可能线上一个点都没有

            expanded_pts = torch.stack(expanded_pts, dim=0).unsqueeze(0)
            result.append(expanded_pts)

        # 处理不定长的 list (由于后续 feature 取 max, 因此直接重复最后一个坐标)
        max_len = max([x.shape[1] for x in result])
        for i, r in enumerate(result):
            if r.shape[1] < max_len:
                result[i] = torch.cat([r, r[:, -1:, :].repeat(1, max_len - r.shape[1], 1)], dim=1)
        return torch.cat(result, dim=0)

    def compute_loss(self, batch_dict, prediction):
        """loss function, combines grasp loss and occupancy loss
        Args:
            batch_dict: gt data
            prediction: net output
        Returns:
            loss (EasyDict): loss values
        """
        loss = EasyDict()

        loss_qual = self.qual_loss_fn(prediction.grasp_label, batch_dict.grasp_label)
        loss_rot = self.rot_loss_fn(prediction.grasp_rotation, batch_dict.grasp_rotation)
        loss_width = self.width_loss_fn(prediction.grasp_width, batch_dict.grasp_width)
        loss.loss_qual = loss_qual.mean()
        loss.loss_rot = loss_rot.mean()
        loss.loss_width = loss_width.mean()
        loss.loss_all = (loss_qual + batch_dict.grasp_label * (1.0 * loss_rot + 16 * loss_width)).mean()

        if self.supervision == "rendered_depth":
            # depth loss
            pixel_pred = prediction.rendered_depth
            img_gt = batch_dict.depth_img[:, 0, :]
            img_gt = img_gt.reshape(img_gt.shape[0], -1)
            pixel_gt = torch.gather(img_gt, 1, prediction.ray_index)
            loss.loss_geo = self.depth_loss_fn(torch.squeeze(pixel_pred),
                                               pixel_gt,
                                               mask=torch.squeeze(prediction.valid_ray_mask)).mean()
            loss.loss_all += loss.loss_geo * self.cfg.geometry_decoder.loss_depth_img_weight

            # Eikonal loss
            if self.training:
                loss.loss_eik = prediction.gradient_error  # same weight as in the paper Ponder
                loss.loss_all += loss.loss_eik * self.cfg.geometry_decoder.loss_eik_weight

            # sdf loss
            n_pts_per_ray = int(prediction.sdf.shape.numel() / prediction.valid_ray_mask.shape.numel())
            valid_pts_mask = prediction.valid_ray_mask.repeat(1, 1, n_pts_per_ray).reshape(-1)
            sdf = prediction.sdf.reshape(-1, 1)[valid_pts_mask]
            approximate_sdf = prediction.approximate_sdf.reshape(-1, 1)[valid_pts_mask]
            is_near_surface = approximate_sdf <= 0.05

            # near surface
            if is_near_surface.sum() > 0:
                loss.loss_near = torch.abs(sdf[is_near_surface] - approximate_sdf[is_near_surface]).sum() / is_near_surface.sum()
                loss.loss_all += loss.loss_near * self.cfg.geometry_decoder.loss_near_weight

            # free space
            if (~is_near_surface).sum() > 0:
                free_space_loss = torch.stack((torch.exp(-5 * sdf[~is_near_surface]) - 1.0,
                                               sdf[~is_near_surface] - approximate_sdf[~is_near_surface]), dim=1).max(1)[0]
                loss.loss_free = free_space_loss.clamp(min=0.0).sum() / (~is_near_surface).sum()
                loss.loss_all += loss.loss_free * self.cfg.geometry_decoder.loss_free_weight

        elif self.supervision == "occupancy":
            loss.loss_geo = self.occ_loss_fn(prediction.occupancy, batch_dict.occupancy).mean()
            loss.loss_all += loss.loss_geo

        elif self.supervision == "none":
            loss.loss_geo = torch.zeros_like(loss.loss_all)
            loss.loss_all += loss.loss_geo

        else:
            raise ValueError("Unknown supervision type: {}".format(self.supervision))

        return loss

    def render_full_image(self, batch_dict, iteration_count, **kwargs):
        # encode tsdf
        feature = self.encoder(batch_dict.tsdf)

        original_size = 0.3
        bounding_box = self.get_bounding_box(self.decoder_padding, original_size)

        camera_extrinsic_matrix = torch.tensor([Transform.from_list(list(batch_dict.camera_extrinsic[:, 0, :][n].cpu())).as_matrix()
                                                for n in range(batch_dict.camera_extrinsic.shape[0])],
                                               device=self.device)

        # render
        render_out = self.renderer.render_full_image(feature,
                                                     batch_dict.camera_intrinsic,
                                                     camera_extrinsic_matrix,
                                                     batch_dict.depth_img,
                                                     bounding_box, original_size, self.decoder_padding,
                                                     iteration_count,
                                                     **kwargs)
        return render_out

    def extract_geometry(self, batch_dict, iteration_count, resolution, **kwargs):
        # encode tsdf
        feature = self.encoder(batch_dict.tsdf)

        original_size = 0.3
        bounding_box = self.get_bounding_box(self.decoder_padding, original_size)

        # render
        render_out = self.renderer.extract_geometry(feature,
                                                    resolution,
                                                    bounding_box, original_size, self.decoder_padding,
                                                    iteration_count,
                                                    **kwargs)
        return render_out

    def predict_grasp(self, batch):
        '''仅预测抓取, 最后输出成 Grasp 位姿
        '''
        prediction = EasyDict()
        with torch.no_grad():
            prediction = self.forward(batch, geometry_branch=False)

        # 结合相机外参 恢复抓取位姿 注意 follow data_generator.py
        batch_size = batch.point_grasp.shape[0]

        # 做一个临时变量 从 yaw 转到四元数 (recall data generator 时的写法)
        predicted_rot = prediction.grasp_rotation
        prediction.grasp_rotation = torch.zeros(batch_size, 4)
        for index in range(batch_size):
            camera_M = Transform(Rotation.from_quat(list(batch.camera_extrinsic[index, 0].cpu())[:4]),
                                 np.array([0, 0, 1])).as_matrix()
            normal = np.linalg.inv(camera_M)[:3, 3]

            R = normal2RM(normal)  # 法向 → 旋转矩阵 (人工固定一个额外的 x 轴)

            ori = R * Rotation.from_euler("z", predicted_rot[index].cpu())

            # ori = Rotation.from_quat(list(batch.camera_extrinsic[index, 0])[:4]) \
            #     * Rotation.from_euler("z", predicted_rot[index].cpu())
            prediction.grasp_rotation[index] = torch.tensor(ori.as_quat())
        return prediction

    def qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none")

    def quat_loss_fn(self, pred, target):
        # for rotation loss 计算四元数差距
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

    def rot_loss_fn(self, pred, target):
        '''只对比旋转角度(余弦相似度, 考虑对称性、周期性)
        '''
        return 1 - torch.cos(2 * (pred - target))  # 如果两个弧度角差别 90° 即 pi/2 那么 loss 会得到 2, 45° ~ loss 1, 30° ~ loss 0.5

    def width_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")

    def occ_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)

    def depth_loss_fn(self, pred, target, mask=None):
        pixel_loss = F.l1_loss(pred, target, reduction="none") * mask
        # return F.mse_loss(pred, torch_resize(target), reduction="none").mean(-1)
        return pixel_loss.view(pixel_loss.shape[0], -1).mean(-1)
