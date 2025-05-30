import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestFusionBlock(nn.Module):
    """
    一个更强大的兴趣融合块，类似于 Transformer 编码器块
    包含交叉注意力 + 前馈网络 + 残差连接 + 层归一化
    """
    def __init__(self, embed_dim, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim or embed_dim * 4
        
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            add_bias_kv=False
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, self.ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: 查询向量 (batch_size, seq_len_q, embed_dim)
            key: 键向量 (batch_size, seq_len_k, embed_dim)
            value: 值向量 (batch_size, seq_len_v, embed_dim)
        """
        # 交叉注意力 + 残差连接 + 层归一化
        attn_output, _ = self.cross_attention(
            query=query, 
            key=key, 
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        return query


class MultiLayerInterestFusion(nn.Module):
    """
    多层兴趣融合模块
    """
    def __init__(self, embed_dim, num_layers=3, num_heads=8, ff_dim=None, dropout=0.1, fusion_strategy='last'):
        super().__init__()
        self.num_layers = num_layers
        self.fusion_strategy = fusion_strategy  # 'last', 'weighted', 'concat'
        
        # 多个融合块
        self.fusion_blocks = nn.ModuleList([
            InterestFusionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 如果使用加权融合
        if fusion_strategy == 'weighted':
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # 如果使用拼接融合
        if fusion_strategy == 'concat':
            self.final_proj = nn.Linear(embed_dim * num_layers, embed_dim)
    
    def forward(self, interests_embed, seq_latents, attn_mask=None, key_padding_mask=None):
        """
        Args:
            interests_embed: 用户兴趣嵌入 (batch_size, 1, embed_dim)
            seq_latents: 序列潜在表示 (batch_size, seq_len, embed_dim)
        """
        outputs = []
        query = interests_embed
        
        for i, fusion_block in enumerate(self.fusion_blocks):
            query = fusion_block(
                query=query,
                key=seq_latents,
                value=seq_latents,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
            outputs.append(query)
        
        # 根据融合策略处理输出
        if self.fusion_strategy == 'last':
            return outputs[-1]
        elif self.fusion_strategy == 'weighted':
            # 加权组合所有层的输出
            weights = F.softmax(self.layer_weights, dim=0)
            weighted_output = sum(w * output for w, output in zip(weights, outputs))
            return weighted_output
        elif self.fusion_strategy == 'concat':
            # 拼接所有层的输出然后投影
            concat_output = torch.cat(outputs, dim=-1)
            return self.final_proj(concat_output)
        else:
            return outputs[-1]


class AdaptiveInterestFusion(nn.Module):
    """
    自适应兴趣融合：根据兴趣内容动态调整注意力权重
    """
    def __init__(self, embed_dim, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 多层融合
        self.fusion_layers = MultiLayerInterestFusion(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            fusion_strategy='weighted'
        )
        
        # 自适应门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, interests_embed, seq_latents):
        """
        Args:
            interests_embed: 用户兴趣嵌入 (batch_size, 1, embed_dim)
            seq_latents: 序列潜在表示 (batch_size, embed_dim)
        """
        # 扩展序列维度以匹配注意力输入
        seq_latents_expanded = seq_latents.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # 多层融合
        fused_interests = self.fusion_layers(interests_embed, seq_latents_expanded)
        fused_interests = fused_interests.squeeze(1)  # (batch_size, embed_dim)
        
        # 自适应门控
        gate = self.gate_network(interests_embed.squeeze(1))  # (batch_size, 1)
        
        # 门控融合
        output = gate * fused_interests + (1 - gate) * seq_latents
        
        return output
