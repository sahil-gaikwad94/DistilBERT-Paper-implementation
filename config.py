# config.py

# Model Hyperparameters
vocab_size = 30522  # Example: Vocabulary size for BERT-base-uncased
embed_dim = 768     # Dimension of token and positional embeddings (Hidden size)
max_seq_len = 128   # Maximum sequence length the model can handle

# BERT (Teacher) Model Specifics
bert_num_layers = 12 # Number of Transformer blocks for the teacher

# DistilBERT (Student) Model Specifics
distilbert_num_layers = 6 # Number of Transformer blocks for the student (typically half of BERT)

num_heads = 12      # Number of attention heads in MultiHeadSelfAttention
ff_dim = embed_dim * 4 # Hidden dimension for the Feed-Forward Network (usually 4x embed_dim)
dropout_rate = 0.1  # Dropout rate for regularization
num_labels = 2      # Number of output classes for classification (e.g., 2 for binary sentiment)

# Training Hyperparameters
temperature = 2.0   # Temperature for knowledge distillation
alpha_distil = 0.5  # Weighting factor for distillation loss
alpha_hard = 0.5    # Weighting factor for hard target loss
learning_rate = 5e-5 # Learning rate for the optimizer
num_epochs = 5      # Number of training epochs
batch_size = 16     # Batch size for training

# Paths
# Path to save/load pre-trained BERT weights (if you have them)
BERT_PRETRAINED_PATH = "./pretrained_bert_weights.pt"
# Path to save the trained DistilBERT model
DISTILBERT_SAVE_PATH = "./distilbert_model.pt"


