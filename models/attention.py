import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    """Hard Sigmoid activation function."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """Hard Swish activation function."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module (CA).
    Captures channel relationships and spatial dependencies with direction-aware
    and position-sensitive attention maps.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.act = h_swish()
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # 1D Global Pooling along height and width
        x_h = self.pool_h(x)  # Shape: (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Shape: (N, C, W, 1)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # Shape: (N, C, H+W, 1)
        
        # Shared 1x1 Conv + BN + Non-linear activation
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # Split back to height and width representations
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Shape: (N, C, 1, W)

        # Generate directional attention weights (sigmoid)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Multiplicative feature enhancement
        out = identity * a_h * a_w

        return out
