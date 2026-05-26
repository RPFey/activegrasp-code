import torch
import torch.nn as nn
import torch.nn.functional as F

# Network Components
# 1. override: conv, conv_stride
# 2. integration: Encoder, Decoder
# 3. net: ConvNet(encoder, decoder, 3-heads)
# 4. only used in vgn
def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )

class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        """
        [Conv3D, relu] * 3 (步长=2)
        """
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        """
        [Conv3D, relu, interpolate] * 3 (上采样)
        """
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = F.interpolate(x, 10)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.interpolate(x, 20)
        x = self.conv3(x)
        x = F.relu(x)

        x = F.interpolate(x, 40)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        """
        [encoder + decoder] → conv_qual, conv_rot, conv_width
        """
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = torch.sigmoid(self.conv_qual(x))
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out
