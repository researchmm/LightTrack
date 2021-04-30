import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_corr(z, x):
    """Pixel-wise correlation (implementation by for-loop and convolution)
    The speed is slower because the for-loop"""
    size = z.size()  # (bs, c, hz, wz)
    CORR = []
    for i in range(len(x)):
        ker = z[i:i + 1]  # (1, c, hz, wz)
        fea = x[i:i + 1]  # (1, c, hx, wx)
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)  # (hz * wz, c)
        ker = ker.unsqueeze(2).unsqueeze(3)  # (hz * wz, c, 1, 1)
        co = F.conv2d(fea, ker.contiguous())  # (1, hz * wz, hx, wx)
        CORR.append(co)
    corr = torch.cat(CORR, 0)  # (bs, hz * wz, hx, wx)
    return corr


def pixel_corr_mat(z, x):
    """Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size()
    z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)


class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PWCA(nn.Module):
    """
    Pointwise Correlation & Channel Attention
    """

    def __init__(self, num_channel, cat=False, CA=True, matrix=False):
        super(PWCA, self).__init__()
        self.cat = cat
        self.CA = CA
        self.matrix = matrix
        if self.CA:
            self.CA_layer = CAModule(channels=num_channel)

    def forward(self, z, x):
        z11 = z[0]
        x11 = x[0]
        # pixel-wise correlation
        if self.matrix:
            corr = pixel_corr_mat(z11, x11)
        else:
            corr = pixel_corr(z11, x11)
        if self.CA:
            # channel attention
            opt = self.CA_layer(corr)
            if self.cat:
                return torch.cat([opt, x11], dim=1)
            else:
                return opt
        else:
            return corr
