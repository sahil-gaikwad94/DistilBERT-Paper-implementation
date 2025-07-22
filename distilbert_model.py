import torch.nn as nn
from modules import Embeddings
from transformer_block import TransformerBlock

class DistilBertModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_layers, num_heads, ff_dim, dropout_rate=0.1, num_labels=2):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim, max_seq_len, dropout_rate)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)

        for block in self.transformer_blocks:
            x = block(x)

        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


