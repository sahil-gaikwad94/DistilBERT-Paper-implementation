import torch
import torch.nn as nn
import torch.nn.functional as F

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        query, key, value = qkv.chunk(3, dim=-1)

        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        context_vector = torch.matmul(attn_weights, value)

        context_vector = context_vector.permute(0, 2, 1, 3).contiguous()
        context_vector = context_vector.view(batch_size, sequence_length, embed_dim)

        output = self.out_proj(context_vector)

        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class AddNorm(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, residual):
        x = residual + self.dropout(x)
        x = self.norm(x)
        return x


