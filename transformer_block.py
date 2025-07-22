import torch.nn as nn
from modules import MultiHeadSelfAttention, FeedForward, AddNorm

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = AddNorm(embed_dim, dropout_rate)

        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm2 = AddNorm(embed_dim, dropout_rate)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(attn_output, x)

        ff_output = self.feed_forward(x)
        x = self.norm2(ff_output, x)

        return x


