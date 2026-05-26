'''
Codes are from:
https://github.com/UT-Austin-RPL/GIGA/blob/main/src/vgn/ConvONets/conv_onet/models/decoder.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .grid_sample_gradfix import grid_sample
from ..latent_space import normalize_coordinate, normalize_3d_coordinate, map2local

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

class FCDecoder(nn.Module):
    '''Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
    dim (int): input dimension
    c_dim (int): dimension of latent conditioned code c
    out_dim (int): dimension of latent conditioned code c
    leaky (bool): whether to use leaky ReLUs
    sample_mode (str): sampling feature strategy, bilinear|nearest
    padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    def __init__(self, dim=3, c_dim=128, out_dim=1, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim

        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = grid_sample(c, vgrid).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = grid_sample(c, vgrid).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        net = self.fc(torch.cat((c, p), dim=2)).squeeze(-1)

        return net


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        in_dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, in_dim=3, c_dim=128,
                 hidden_size=256,
                 n_blocks=5,
                 out_dim=1,
                 leaky=False,
                 sample_mode='bilinear',
                 padding=0.1,
                 concat_feat=False,
                 expand_sample_pts=0,  # concat 输入的多点 feature
                 with_ray_feature=False,  # 压缩最后一系列点的 feature
                 for_grasp=False,
                 no_xyz=False):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:  # 默认进
            c_dim *= 3
        self.expand_sample_pts = expand_sample_pts
        self.with_ray_feature = with_ray_feature
        self.sampled_pts = 1 + expand_sample_pts + with_ray_feature
        self.for_grasp = for_grasp
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = True if in_dim == 0 else False
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim*self.sampled_pts, hidden_size) for i in range(n_blocks)  # 一系列从 c_dim 维到 hidden_size 维的线性映射, 在不同阶段加入 hidden_size
            ])

        if not self.no_xyz:
            self.fc_p = nn.Linear(in_dim, hidden_size)  # 从 input 映射到 hidden_size 维, 默认是 32 维

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)  # 两层 MLP+ReLU, 加了残差连接
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        """输入平面坐标 p, 从特征平面 c 中采样
        """
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # [-0.5, 0.5] -> [0, 1]
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        # ↓ vgrid 平面坐标, 范围(-1, 1)
        # 原本是双线性插值来调整 tensor 的大小, 但这里 vgrid 的维度相当于只有一个点, 所以是采样
        c = grid_sample(c, vgrid).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        """输入三维坐标 p, 从特征体 c 中采样
        """
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = grid_sample(c, vgrid).squeeze(-1).squeeze(-1)
        return c

    def _sample_feature(self, p, c_plane):
        """输入坐标 p, 从 c_plane.keys() 中采样, 最后进行 concat 或者相加
        support multiple points: p[Batch, N_pts, 3]
        """
        assert p.shape[-1] == 3, 'p should be 3D coordinates'
        if self.c_dim != 0:  # 默认 96
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if 'grid' in plane_type:
                    c = self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
                if 'xy' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
                if 'yz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if 'grid' in plane_type:
                    c += self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                if 'xy' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                if 'yz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c = c.transpose(1, 2)
        return c

    def _compute(self, p, c):
        '''p [batch_size, 1, 3];  c [batch_size, 1, c_dim*3=96 if concat]
        '''
        if self.no_xyz:  # 如果没有额外输入, 则构造一个全零的
            net = torch.zeros(c.size(0), c.size(1), self.hidden_size).to(c.device)
        else:
            p = p.float()
            net = self.fc_p(p)  # 一个全连接层, 从 input 映射到 hidden_size 维

        for i in range(self.n_blocks):  # 默认 5
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)  # 全连接层, 加入 feature

            net = self.blocks[i](net)  # 两层 MLP+ReLU, 加了残差连接

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out

    def forward(self, pos, c_plane, input=None, **kwargs):
        """前向过程 divide into _sample_feature and _compute
        pos ∈ (-0.5, 0.5), [batch_size, 1, 3]
        c: dict['xz', 'xy', 'yz' or 'grid']
        """
        c = self._sample_feature(pos, c_plane)
        if self.for_grasp:  # 多个点输出一个 feature
            c = c.reshape(c.shape[0], 1, -1)
            if self.with_ray_feature:  # 将 ray feature 取 max 后 concat 到 c 中
                batch_size = pos.shape[0]
                assert c.shape[-1] > self.c_dim*(self.sampled_pts-1), "at least one point on the ray"
                c_on_ray = c[..., self.c_dim*(self.sampled_pts-1):].reshape(batch_size, 1, -1, self.c_dim)
                max_c_on_ray = torch.max(c_on_ray, dim=2)[0]
                c = torch.cat([c[..., :self.c_dim*(self.sampled_pts-1)], max_c_on_ray], dim=-1)  # ray 上的只取 max

        out = self._compute(input, c)
        return out

    def gradient(self, x, c_planes):
        """计算输出向量对于 x 的梯度
        """
        x.requires_grad_(True)
        for k, v in c_planes.items():
            v.requires_grad_(True)

        c = self._sample_feature(x, c_planes)
        y = self._compute(x, c).unsqueeze(-1)

        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,  # [x, c],
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)  # tuple, [0] 表示对 x 的导数, [1] 表示对 c 的导数
        return gradients[0].unsqueeze(1)

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        """【Not used】只比 _compute 少了一个 self.no_xyz 的判断
        """
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


class PatchLocalDecoder(nn.Module):
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        # self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)

    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = grid_sample(c, vgrid).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = grid_sample(c, vgrid).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalPointDecoder(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val']**2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        # p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6)**2
            weight = (dist / self.var).exp()  # Guassian kernel
        else:
            weight = 1 / ((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6)

        # weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
            if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
