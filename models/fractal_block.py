import torch
import torch.nn as nn

class FractalDownsamplingBlock(nn.Module):
    """
    Fractal Downsampling Block (FDB).
    Replaces standard max/avg pooling with a multi-level branching convolution structure.
    Extracts features at different scales and receptive fields, reducing feature 
    redundancy during spatial downsampling.
    """
    def __init__(self, in_channels, out_channels):
        super(FractalDownsamplingBlock, self).__init__()
        
        # Branch 1: Single-stage downsampling using 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Two-stage downsampling (1x1 conv for channel adaptation -> 3x3 conv stride 2)
        # Captures deeper feature hierarchy before downsampling
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: Average pooling for spatial reduction -> 3x3 conv for feature refinement
        # Preserves local spatial average while extracting features
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer: Concatenates features from all branches and projects them back to out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate branches along the channel dimension
        merged = torch.cat([out1, out2, out3], dim=1)
        
        # Fuse multi-scale features
        return self.fusion(merged)
