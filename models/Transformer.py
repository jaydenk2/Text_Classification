import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads):
        super().__init__()
        self.head_dim = embedding_dim // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, mask=None):
        batch_size = q.shape[0]
        Q = self.w_q(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
             scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        return self.out_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiheadAttention(embedding_dim, n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = PositionWiseFFN(embedding_dim, d_ff=embedding_dim*4)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, n_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(embedding_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embedding_dim, n_heads) for _ in range(num_layers)]
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(embedding_dim * 3, embedding_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(embedding_dim, num_classes)
        # )
        
        self.classifier = nn.Sequential( # USING Element-wise Absolute Difference for second transformer model
        nn.Linear(embedding_dim * 4, embedding_dim), # Changed from 3 to 4
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(embedding_dim, num_classes)
    )

    def make_src_mask(self, src): # masked attention
        mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return mask

    def forward_one(self, x, mask):
        x = self.embedding(x)
        x = self.position_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1)
        x = x * mask_expanded.float()
        sum_embeddings = x.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask
        return pooled

    # def forward(self, q1, q2):
    #     mask1 = self.make_src_mask(q1)
    #     mask2 = self.make_src_mask(q2)
    #     u = self.forward_one(q1, mask1)
    #     v = self.forward_one(q2, mask2)
    #     diff = torch.abs(u - v)
        
    #     combined = torch.cat((u, v, diff), dim=1)
    #     return self.classifier(combined)
    
    
    def forward(self, q1, q2): # for second transformer model with element-wise product
        mask1 = self.make_src_mask(q1)
        mask2 = self.make_src_mask(q2)
        u = self.forward_one(q1, mask1)
        v = self.forward_one(q2, mask2)
        
        diff = torch.abs(u - v)
        prod = u * v 
        combined = torch.cat((u, v, diff, prod), dim=1) 
        return self.classifier(combined)