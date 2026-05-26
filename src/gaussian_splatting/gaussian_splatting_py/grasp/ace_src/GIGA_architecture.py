import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F

from .decoder_module import decoder
from .encoder_module import voxels

# import from ConvONet
# Decoder dictionary
ConvONet_decoder_dict = {
    # 'simple_fc': decoder.FCDecoder,
    # 'simple_local': decoder.LocalDecoder,
    # 'simple_local_crop': decoder.PatchLocalDecoder,
    # 'simple_local_point': decoder.LocalPointDecoder
}
# Encoder dictionary
ConvONet_encoder_dict = {
    # 'pointnet_local_pool': pointnet.LocalPoolPointnet,
    # 'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    # 'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    # 'voxel_simple_local': voxels.LocalVoxelEncoder,
}


def get_GIGA_model(device=None, dataset=None, **kwargs):
    '''
    Return the Occupancy Network model. using GIGA original config.

    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    encoder_type = 'voxel_simple_local'
    decoder_type = 'simple_local'
    encoder_kwargs = {'plane_type': ['xz', 'xy', 'yz'],
                      'plane_resolution': 40, 'unet': True,
                      'unet_kwargs':
                      {'depth': 3, 'merge_mode': 'concat', 'start_filts': 32}}
    decoder_kwargs = {'dim': 3, 'sample_mode': 'bilinear',
                      'hidden_size': 32, 'concat_feat': True}
    padding = 0
    c_dim = 32

    # for pointcloud_crop (removed)
    # local positional encoding (removed)

    is_decoder_tsdf = True  # for all models
    tsdf_only = False  # for GIGAGeo
    detach_tsdf = False  # for GIGADetach

    # construct decoders, for grasp quality, rotation, width, TSDF
    # LocalDecoder from decoder.py
    # the only difference is the output channel
    if tsdf_only:
        decoders = []
    else:  # True
        decoder_qual = decoder.LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoder_rot = decoder.LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=4,
            **decoder_kwargs
        )
        decoder_width = decoder.LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders = [decoder_qual, decoder_rot, decoder_width]
    if is_decoder_tsdf or tsdf_only:  # True
        decoder_tsdf = decoder.LocalDecoder(
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)

    # construct encoder
    # LocalVoxelEncoder from voxels.py
    if encoder_type == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder_type is not None:  # True
        encoder = voxels.LocalVoxelEncoder(
            c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if tsdf_only:
        model = ConvolutionalOccupancyNetworkGeometry(
            decoder_tsdf, encoder, device=device
        )
    else:  # True
        model = ConvolutionalOccupancyNetwork(
            decoders, encoder, device=device, detach_tsdf=detach_tsdf
        )

    return model


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_width = decoders[2].to(device)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        # feature = self.query_feature(p, c)
        # qual, rot, width = self.decode_feature(p, feature)
        qual, rot, width = self.decode(p, c)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()  # 分离出 tensor, 不参与梯度计算
                    # 在这里指的是 GIGADetach 模式下，隐空间的编码 c 不再更新
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = - qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            # print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, width_out

    def compute_loss(self, y_pred, y):
        """
        loss function, combines grasp loss and occupancy loss
        """
        label_pred, rotation_pred, width_pred, occ_pred = y_pred
        label, rotations, width, occ = y
        loss_qual = self.__qual_loss_fn(label_pred, label)
        loss_rot = self.__rot_loss_fn(rotation_pred, rotations)
        loss_width = self.__width_loss_fn(width_pred, width)
        loss_occ = self.__occ_loss_fn(occ_pred, occ)
        loss = loss_qual + label * (loss_rot + 0.01 * loss_width) + loss_occ
        loss_dict = {'loss_qual': loss_qual.mean(),
                     'loss_rot': loss_rot.mean(),
                     'loss_width': loss_width.mean(),
                     'loss_occ': loss_occ.mean(),
                     'loss_all': loss.mean()}
        return loss.mean(), loss_dict

    def __qual_loss_fn(self, pred, target):
        # start with "_" means private function, can not used outside this file
        return F.binary_cross_entropy(pred, target, reduction="none")

    def __quat_loss_fn(self, pred, target):
        # for rotation loss
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

    def __rot_loss_fn(self, pred, target):
        loss0 = self.__quat_loss_fn(pred, target[:, 0])
        loss1 = self.__quat_loss_fn(pred, target[:, 1])
        return torch.min(loss0, loss1)

    def __width_loss_fn(self, pred, target):
        return F.mse_loss(40 * pred, 40 * target, reduction="none")

    def __occ_loss_fn(self, pred, target):
        # size: (batch_size, 2048)
        # target: sparse 0/1 occupancy  # from where?
        return F.binary_cross_entropy(pred, target, reduction="none").mean(-1)
        # size: (batch_size)


class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder_tsdf = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode_occ(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
