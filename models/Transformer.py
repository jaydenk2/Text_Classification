import torch
import torch.nn as nn
import numpy as np

# referenced lecture 22 from course material
# helper functions as described in lecture

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, mode, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.mode = mode
        self.embedding_dim = embedding_dim
        if mode == 'learned':
            self.embedding = nn.Embedding(max_len, embedding_dim)
    
    def forward(self, x):
        if self.mode is None:
            p = torch.zeros_like(x)
        elif self.mode == 'sinusoidal':
            d = self.embedding_dim
            L = x.size(1) 
            p = torch.zeros(L, d, device=x.device)
            for pos in range(L):
                for i in range(d):
                    if i % 2 == 0:
                        p[pos, i] = np.sin(pos / (10000**(i/d)))
                    else:
                        p[pos, i] = np.cos(pos / (10000**(i/d)))
            p = p.unsqueeze(0).expand(x.size(0), -1, -1)
        elif self.mode == 'learned':
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            p = self.embedding(pos)
        else:
            raise Exception("mode should be None, 'sinusoidal', or 'learned'")
        return p.float()

class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_heads):
        super(MultiheadAttention, self).__init__()
        multi_embedding_dim = embedding_dim // n_heads
        multi_hidden_dim = hidden_dim // n_heads
        self.n_heads = n_heads
        
        self.Q_layers = nn.ModuleList([nn.Linear(hidden_dim, multi_hidden_dim) for _ in range(n_heads)])
        self.K_layers = nn.ModuleList([nn.Linear(hidden_dim, multi_hidden_dim) for _ in range(n_heads)])
        self.V_layers = nn.ModuleList([nn.Linear(embedding_dim, multi_embedding_dim) for _ in range(n_heads)])
        
        self.attn_head = ScaledDotProductAttention()
        self.output_linear = nn.Linear(n_heads * multi_embedding_dim, embedding_dim)

    def forward(self, Q, K, V, mask=None):
        self_attn_results = torch.cat([
            self.attn_head(self.Q_layers[i](Q), self.K_layers[i](K), self.V_layers[i](V), mask=mask) 
            for i in range(self.n_heads)
        ], dim=-1)
        
        multi_attn_output = self.output_linear(self_attn_results)
        return multi_attn_output

class FullEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, position_mode, n_heads):
        super(FullEncoder, self).__init__()
        self.position_encoding = PositionalEncoding(position_mode, embedding_dim)
        self.multi_head_attention = MultiheadAttention(embedding_dim, hidden_dim, n_heads)
        self.ffn = PositionWiseFFN(embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc_Q = nn.Linear(embedding_dim, hidden_dim)
        self.fc_K = nn.Linear(embedding_dim, hidden_dim)
        self.fc_V = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        z = self.embedding(x)
        z = z + self.position_encoding(z)
        Q = self.fc_Q(z)
        K = self.fc_K(z)
        V = self.fc_V(z)
        attn_output = self.multi_head_attention(Q, K, V)
        attn_output = self.layer_norm(attn_output + z)
        ffn_output = self.ffn(attn_output)
        encoder_result = self.layer_norm(ffn_output + attn_output)
        return self.activation(encoder_result)

class TransformerClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim, num_classes, n_heads=4):
        super().__init__()
        self.encoder = FullEncoder(num_embeddings, embedding_dim, hidden_dim, 'sinusoidal', n_heads)
        self.fc_out = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.fc_out(pooled)