import torch
import torch.nn as nn
import math

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, num_dims, max_freq=10000):
        """
        Args:
            d_model: 输出的embedding维度
            num_dims: 坐标系维度（如3为x yz，4为xyzt等）
            max_freq: 最大频率，用于计算frequency
        """
        super().__init__()
        self.d_model = d_model
        self.num_dims = num_dims
        
        # 每个维度分配的embedding大小（向下取整）
        self.d_per_dim = (d_model // num_dims) // 2 * 2
        # 实际使用的总维度
        self.used_dims = self.d_per_dim * num_dims
        # 需要补零的维度
        self.padding_dims = d_model - self.used_dims
        
        # 创建频率项
        half_dim = self.d_per_dim // 2
        freqs = torch.exp(
            -math.log(max_freq) * torch.arange(0, half_dim) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, coords):
        """
        Args:
            coords: [..., num_dims] 最后一维是坐标
        Returns:
            [..., d_model] 位置编码
        """
        *batch_shape, num_coords = coords.shape
        assert num_coords == self.num_dims, f"Expected {self.num_dims} coordinates, got {num_coords}"
        
        # 计算所有维度的角度: [..., num_dims, half_dim]
        angles = coords.unsqueeze(-1) * self.freqs
        
        # 计算sin和cos并交错排列: [..., num_dims, d_per_dim]
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        enc = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)
        
        # 展平所有维度的编码: [..., num_dims * d_per_dim]
        output = enc.flatten(-2)
        
        # 如果需要，在末尾补零
        if self.padding_dims > 0:
            padding = torch.zeros(*batch_shape, self.padding_dims, 
                                 device=output.device, dtype=output.dtype)
            output = torch.cat([output, padding], dim=-1)
        
        return output