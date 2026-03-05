import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureProjection(nn.Module):
    def __init__(
        self,
        input_dim,
        feat_batch,
        batch_output_dim,
        batch_hidden_dim=None,
        num_layers=2,
        dropout=0.1,
        activation='relu'
    ):
        """
        特征映射模块
        
        Args:
            input_dim: 输入特征维度
            feat_batch: 每个batch包含的特征数量
            batch_output_dim: 每个batch的输出特征维度
            batch_hidden_dim: 每个batch的隐层维度，默认为feat_batch和batch_output_dim的平均值
            num_layers: 线性层数量
            dropout: dropout比率
            activation: 激活函数类型，支持'relu'、'gelu'、'tanh'
        """
        super().__init__()
        
        # 获得feat_batch数目
        self.feat_batch = feat_batch
        self.feat_batch_count = (input_dim + feat_batch - 1) // feat_batch
        self.last_batch_size = input_dim % feat_batch if input_dim % feat_batch != 0 else feat_batch
        
        # 获得每个batch需要的隐层维度
        if batch_hidden_dim is None:
            batch_hidden_dim = (feat_batch + batch_output_dim) // 2
            
        self.batch_output_dim = batch_output_dim
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        act_fn = activation_map.get(activation.lower(), nn.ReLU())
        
        # 为每个特征批次创建独立的线性层
        self.input_projs = nn.ModuleList()
        for i in range(self.feat_batch_count):
            # 确定当前批次的实际输入大小
            batch_input_size = self.last_batch_size if i == self.feat_batch_count - 1 else feat_batch
            
            # 如果num_layers为1，直接映射到输出维度
            if num_layers == 1:
                self.input_projs.append(nn.Linear(batch_input_size, batch_output_dim))
            else:
                self.input_projs.append(nn.Linear(batch_input_size, batch_hidden_dim))
        
        # 构建后续的转换网络
        if num_layers > 1:
            self.transforms = nn.ModuleList([self._build_transform(batch_hidden_dim, batch_output_dim, 
                                                                   num_layers-1, dropout, act_fn) 
                                             for _ in range(self.feat_batch_count)])
        else:
            self.transforms = None
    
    def _build_transform(self, input_dim, output_dim, num_layers, dropout, act_fn):
        """构建转换网络"""
        layers = []
        
        for i in range(num_layers):
            # 加入激活函数和dropout
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            
            # 添加线性层，映射到下一维度
            next_dim = output_dim if i == num_layers - 1 else input_dim
            layers.append(nn.Linear(input_dim, next_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, ..., input_dim]
        Returns:
            输出张量 [batch_size, ..., feat_batch_count, batch_output_dim]
        """
        # 获取batch维度等信息
        prefix_shape = x.shape[:-1]
        
        # 分割输入特征到不同批次
        batch_outputs = []
        for i in range(self.feat_batch_count):
            start_idx = i * self.feat_batch
            end_idx = min(start_idx + self.feat_batch, x.shape[-1])
            
            # 提取当前批次的特征
            batch_input = x[..., start_idx:end_idx]
            
            # 通过对应的映射层
            batch_output = self.input_projs[i](batch_input)
            
            # 如果有后续转换，应用转换
            if self.transforms is not None:
                batch_output = self.transforms[i](batch_output)
                
            batch_outputs.append(batch_output)
        
        # 堆叠所有批次输出
        return torch.stack(batch_outputs, dim=-2)