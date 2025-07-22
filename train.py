import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import (
    vocab_size, embed_dim, max_seq_len, bert_num_layers,
    distilbert_num_layers, num_heads, ff_dim, dropout_rate,
    num_labels, temperature, alpha_distil, alpha_hard,
    learning_rate, num_epochs, batch_size, BERT_PRETRAINED_PATH, DISTILBERT_SAVE_PATH
)
from bert_model import BertEncoder
from distilbert_model import DistilBertModel
from loss import DistillationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

dummy_texts = [
    "This is a positive sentence and I love it.",
    "I hate this movie, it was terrible.",
    "What a wonderful day! The sun is shining.",
    "This is absolutely terrible, I am so disappointed.",
    "I love this product, it's the best I've ever used.",
    "Not good, very bad experience.",
    "Highly recommend this, amazing!",
    "Worst service ever, never again."
]
dummy_labels = [1, 0, 1, 0, 1, 0, 1, 0]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

dataset = SimpleTextDataset(dummy_texts, dummy_labels, tokenizer, max_seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

bert_teacher_model = BertEncoder(vocab_size, embed_dim, max_seq_len, bert_num_layers, num_heads, ff_dim, dropout_rate).to(device)
try:
    print("Teacher model (BertEncoder) initialized. (No pre-trained weights loaded in this example)")
except FileNotFoundError:
    print(f"Warning: Pre-trained BERT weights not found at {BERT_PRETRAINED_PATH}. Teacher model will use random initialization.")
    print("For proper distillation, a pre-trained BERT model is required.")
bert_teacher_model.eval()

distilbert_student_model = DistilBertModel(vocab_size, embed_dim, max_seq_len, distilbert_num_layers, num_heads, ff_dim, dropout_rate, num_labels).to(device)

criterion = DistillationLoss(temperature, alpha_distil, alpha_hard)

optimizer = torch.optim.AdamW(distilbert_student_model.parameters(), lr=learning_rate)

print("\nStarting training...")
distilbert_student_model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_output = bert_teacher_model(input_ids)
            teacher_logits = nn.Linear(embed_dim, num_labels).to(device)(teacher_output[:, 0, :])

        student_logits = distilbert_student_model(input_ids)

        loss = criterion(student_logits, teacher_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {total_loss/len(dataloader):.4f}")

print("Training complete! Saving student model...")
torch.save(distilbert_student_model.state_dict(), DISTILBERT_SAVE_PATH)
print(f"Student model saved to {DISTILBERT_SAVE_PATH}")

print("\nStarting inference example...")
distilbert_student_model.eval()

example_texts = [
    "This product is absolutely fantastic!",
    "I am so disappointed with the quality.",
    "Neutral statement.",
    "Best purchase ever."
]

for i, text in enumerate(example_texts):
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded_input["input_ids"].to(device)

    with torch.no_grad():
        output_logits = distilbert_student_model(input_ids)
        probabilities = F.softmax(output_logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    print(f"\nExample {i+1} Text: \n{text}")
    print(f"Predicted Probabilities: {probabilities.cpu().numpy()}")
    print(f"Predicted Class (0: Negative, 1: Positive): {predicted_class}")

print("Inference example complete.")


