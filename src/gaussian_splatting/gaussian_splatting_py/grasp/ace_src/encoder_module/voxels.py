'''
Codes are from:
https://github.com/autonomousvision/convolutional_occupancy_networks/blob/master/src/encoder/voxels.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from .unet import UNet
from .unet3d import UNet3D
from ..latent_space import coordinate2index, normalize_coordinate, normalize_3d_coordinate


class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.
    输入 voxel, 输出 fea 字典, 相应包含 grid 或者三个平面的特征
    每个平面是 [batch_size, c_dim, plane_resolution, plane_resolution]

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c                         -> 隐空间
        hidden_dim (int): hidden dimension of the network               -> ???
        unet (bool): weather to use U-Net                               -> whether hhh
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature    -> 压缩到平面特征，平面的分辨率
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN       -> 第一层卷积核大小
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
                                                                        -> ???
    '''

    def __init__(self, dim=3, c_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.c_dim = c_dim

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution

        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        '''p(batch_size, n_voxel, 3) n_voxel = 40*40*40
           c(batch_size, n_voxel, c_dim)
        '''
        # acquire indices of features in plane
        # 3 对应三个轴, 抽取对应平面, 规范化到 [0, 1]
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)

        # xy(batch_size, n_voxel, 2)
        # 按照规则给 index
        index = coordinate2index(xy, self.reso_plane)
        # index(batch_size, 1, n_voxel)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)  # TODO why c.new_zeros?
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        # 插值+平均
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)  # e.g. [batch_size, c_dim=32, 40, 40] -> [batch_size, c_dim=32, 40, 40]

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def forward(self, x):
        # x: (batch_size, depth, height, width) eg 32*40*40*40
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates 生成一个特征空间
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        # 相当于把这个 linspace 向量拷贝扩展到整个空间 (batch_size, depth, height, width)
        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4)  # (batch_size, depth, height, width, 3)  可以形象地理解成三个梯度指向正交的长方体（每个元素是标量）
        p = p.view(batch_size, n_voxel, -1)  # (batch_size, n_voxel, 3) eg 32*64000*3 把每个长方体拆成一维向量，三个长方体
        # 这里的结果相当于[012012012...][000111222...]几种不同的索引（注意转到三维空间）

        # Acquire voxel-wise feature
        x = x.unsqueeze(1)  # 添加了一个大小为 1 的维度(作为 channel)，在第二个维度上 (batch_size, depth, height, width)->(batch_size, 1, depth, height, width)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)  # channel 1->c_dim, 剩下的维度由卷积核等控制（实际上也要一一对应，维度和voxel不变）
        c = c.permute(0, 2, 1)  # 维度交换 (batch_size, -1, c_dim)

        # p 相当于空的特征空间索引，c 是 voxel 经过一次 3D 卷积、relu 之后得到的特征
        # e.g. batch_size=32, self.c_dim=32, voxel=40*40*40
        # p: [32, 64000, 3] c: [32, 64000, 32]
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')
                # e.g. [batch_size, c_dim=32, 40, 40]

        return fea


class VoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c
