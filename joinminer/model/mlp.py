import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    通用输出转换模块，用于将输入embed转换为指定维度的输出
    
    Args:
        input_dim (int): 输入特征维度
        output_dim (int): 输出特征维度
        hidden_dims (list, optional): 隐藏层维度列表，默认None表示使用一个中间层
        dropout (float, optional): dropout比率，默认0.1
        activation (str, optional): 激活函数类型，支持'relu'、'gelu'、'tanh'等，默认'relu'
        final_activation (str, optional): 最后一层的激活函数，支持None、'relu'、'sigmoid'等，默认'relu'
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=None,
        dropout=0.1,
        activation='relu',
        final_activation='relu'
    ):
        super().__init__()
        
        # 如果未指定hidden_dims，使用一个中间层
        if hidden_dims is None:
            # hidden_dims = [(input_dim + output_dim) // 2]
            hidden_dims = []
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leakyrelu': nn.LeakyReLU(),
            None: nn.Identity()
        }
        
        act_fn = activation_map.get(activation.lower() if activation else None, nn.ReLU())
        final_act_fn = activation_map.get(final_activation.lower() if final_activation else None, nn.Identity())
        
        # 构建网络层
        layers = []
        current_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            # 线性层
            layers.append(nn.Linear(current_dim, hidden_dim))
                
            # 激活函数
            layers.append(act_fn)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
            
        # 输出层激活函数
        if final_activation is not None:
            layers.append(final_act_fn)
        
        self.transform = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, ..., input_dim]
            
        Returns:
            输出张量 [batch_size, ..., output_dim]
        """
        return self.transform(x)