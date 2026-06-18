import torch
import torch.nn as nn
from .attention import CoordinateAttention
from .fractal_block import FractalDownsamplingBlock

class DualPathEnhancementBlock(nn.Module):
    """
    Dual-Path Enhancement Block (DEB).
    Creates two parallel paths:
      Path 1 (Residual Learning Path): Focuses on residual learning and mapping identity.
      Path 2 (Dense Feature Reuse Path): Focuses on reusing shallow and deep features to prevent gradient vanishing.
    Outputs are concatenated and fused.
    """
    def __init__(self, in_channels, out_channels):
        super(DualPathEnhancementBlock, self).__init__()
        
        # --- Path 1: Residual Learning Path ---
        self.path1_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.path1_bn1 = nn.BatchNorm2d(out_channels)
        self.path1_relu = nn.ReLU(inplace=True)
        self.path1_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.path1_bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for Path 1 if channels don't match
        if in_channels != out_channels:
            self.path1_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.path1_shortcut = nn.Identity()
            
        # --- Path 2: Dense Feature Reuse Path ---
        # First layer extracts mid-level features
        mid_channels = out_channels // 2
        self.path2_conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.path2_bn1 = nn.BatchNorm2d(mid_channels)
        
        # Second layer concatenates block input and first-layer output
        self.path2_conv2 = nn.Conv2d(in_channels + mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.path2_bn2 = nn.BatchNorm2d(out_channels)
        
        # --- Fusion Layer ---
        # Fuses Path 1 (out_channels) + Path 2 (out_channels) back to out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Forward Path 1 (Residual Path)
        out_path1 = self.path1_conv1(x)
        out_path1 = self.path1_bn1(out_path1)
        out_path1 = self.path1_relu(out_path1)
        out_path1 = self.path1_conv2(out_path1)
        out_path1 = self.path1_bn2(out_path1)
        out_path1 = self.path1_relu(out_path1 + self.path1_shortcut(x))
        
        # Forward Path 2 (Dense Feature Reuse Path)
        p2_f1 = self.path1_relu(self.path2_bn1(self.path2_conv1(x)))
        p2_merged = torch.cat([x, p2_f1], dim=1)
        out_path2 = self.path1_relu(self.path2_bn2(self.path2_conv2(p2_merged)))
        
        # Combine Paths
        combined = torch.cat([out_path1, out_path2], dim=1)
        return self.fusion(combined)


class UCDFNet(nn.Module):
    """
    Universal Color Dual-Path Fractal Network (UC-DFNet) for Color Image Steganalysis.
    
    Structure:
      - Front-end Conv (extracts high-frequency noise and spatial features)
      - Stage 1: DEB -> Coordinate Attention -> FDB (downsample 2x)
      - Stage 2: DEB -> Coordinate Attention -> FDB (downsample 2x)
      - Stage 3: DEB -> Coordinate Attention -> FDB (downsample 2x)
      - Classification Head: Global Average Pooling -> FC -> Softmax output (logits)
    """
    def __init__(self, num_classes=2):
        super(UCDFNet, self).__init__()
        
        # Front-end Preprocessing layer to extract initial color noise maps
        self.front_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 32 -> 64 channels, downsample 256x256 -> 128x128
        self.stage1_deb = DualPathEnhancementBlock(32, 64)
        self.stage1_attn = CoordinateAttention(64, 64)
        self.stage1_fdb = FractalDownsamplingBlock(64, 64)
        
        # Stage 2: 64 -> 128 channels, downsample 128x128 -> 64x64
        self.stage2_deb = DualPathEnhancementBlock(64, 128)
        self.stage2_attn = CoordinateAttention(128, 128)
        self.stage2_fdb = FractalDownsamplingBlock(128, 128)
        
        # Stage 3: 128 -> 256 channels, downsample 64x64 -> 32x32
        self.stage3_deb = DualPathEnhancementBlock(128, 256)
        self.stage3_attn = CoordinateAttention(256, 256)
        self.stage3_fdb = FractalDownsamplingBlock(256, 256)
        
        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Front-end feature extraction
        x = self.front_conv(x)
        
        # Stage 1
        x = self.stage1_deb(x)
        x = self.stage1_attn(x)
        x = self.stage1_fdb(x)
        
        # Stage 2
        x = self.stage2_deb(x)
        x = self.stage2_attn(x)
        x = self.stage2_fdb(x)
        
        # Stage 3
        x = self.stage3_deb(x)
        x = self.stage3_attn(x)
        x = self.stage3_fdb(x)
        
        # Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        return logits
        
    def get_stage_features(self, x):
        """Helper function for testing and visualization to extract intermediate feature maps."""
        feats = {}
        
        x = self.front_conv(x)
        feats['front_conv'] = x
        
        x = self.stage1_deb(x)
        feats['stage1_deb'] = x
        x = self.stage1_attn(x)
        feats['stage1_attn'] = x
        x = self.stage1_fdb(x)
        feats['stage1_fdb'] = x
        
        x = self.stage2_deb(x)
        feats['stage2_deb'] = x
        x = self.stage2_attn(x)
        feats['stage2_attn'] = x
        x = self.stage2_fdb(x)
        feats['stage2_fdb'] = x
        
        x = self.stage3_deb(x)
        feats['stage3_deb'] = x
        x = self.stage3_attn(x)
        feats['stage3_attn'] = x
        x = self.stage3_fdb(x)
        feats['stage3_fdb'] = x
        
        return feats
