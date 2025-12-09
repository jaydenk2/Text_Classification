import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = int (kernel_size / 2)  # to maintain the same length after conv
        self.block = nn.Sequential(
            # no need for 2d since input is text
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride = 1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride= 2)
        )
    def forward(self, x):
        return self.block(x)
    
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.block1 = CNNBlock(in_channels=embed_dim, out_channels=64, kernel_size=3)
        self.block2 = CNNBlock(in_channels=64, out_channels=128, kernel_size=3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1) # permuation for conv1d to batch, channels, seq_len
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x).squeeze(-1)
        z = self.fc(x)
        return z
        
    
       