import torch
from torch import nn, einsum
from einops import rearrange
from models import Dilation2d, Erosion2d
from self_attention import Self_Attn

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class ConvErosion_Dilation_Block(nn.Module):
    def __init__(self, dim, output_dim, kernel_size_conv, kernel_size_diaero):
        super().__init__()
        self.conv_dia_block = nn.Sequential(
            nn.Conv2d(dim, output_dim, kernel_size_conv, 1, 1),
            Dilation2d(output_dim, output_dim, kernel_size=kernel_size_diaero)
        )

        self.conv_ero_block = nn.Sequential(
            nn.Conv2d(dim, output_dim, kernel_size_conv, 1, 1),
            Erosion2d(output_dim, output_dim, kernel_size=kernel_size_diaero) 
        )

    def forward(self, x):
        x = torch.cat((self.conv_dia_block(x), self.conv_ero_block(x)), dim=1)
        return x


class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 4, dim_head = 64, kernel_size=3, 
    q_stride=1, k_stride=1, v_stride=1, dropout = 0.2):

        super().__init__()
        self.img_size = img_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.conv_erosion_dilation1 = ConvErosion_Dilation_Block(dim = dim, output_dim = dim//2, kernel_size_conv = 3, kernel_size_diaero = 5)
        self.conv_erosion_dilation2 = ConvErosion_Dilation_Block(dim = dim, output_dim = dim//2, kernel_size_conv = 5, kernel_size_diaero = 3)
        self.self_attention = Self_Attn(
            in_dim = dim, kernel_size = kernel_size, 
            q_stride=q_stride, k_stride=k_stride, v_stride=v_stride)


    def forward(self, x):
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)

        # First ConvErosion_Dilation_Block:
        x = self.conv_erosion_dilation1(x)

        # Attention:
        out = self.self_attention(x)

        # Second ConvErosion_Dilation_Block:

        x = self.conv_erosion_dilation2(out)
        out = rearrange(out, 'b h n d -> b (n d) h')
        return out
