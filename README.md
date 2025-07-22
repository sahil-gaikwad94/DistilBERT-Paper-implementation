# DistilBERT-Paper-implementation

# 🦄 DistilBERT Paper Implementation — Tiny Transformers, Big Results!

Welcome to the *unofficially-official* playground for DistilBERT!  
If you’ve ever thought: “What if BERT had a little sibling who’s faster, lighter, and still aces the test?” — you’re in the right place!

---

## 🚀 What is This?

This repo is a from-scratch, PyTorch-powered implementation of the [DistilBERT](https://arxiv.org/abs/1910.01108) architecture, just like the famous paper by Victor Sanh et al.  
We focus on clarity, modularity, and that “aha!” moment for anyone curious about transformer distillation.

- **No black-box magic:** Every block is transparent and hackable.
- **Distillation in action:** See how knowledge is handed down from BERT (the wise teacher) to DistilBERT (the eager student).
- **Plug-n-play modules:** Embeddings, attention, transformer blocks, it’s all here!

---

## 🧰 Repo Structure (TL;DR)

| File/Dir               | What’s Inside                                                              |
|------------------------|----------------------------------------------------------------------------|
| `distilbert_model.py`  | The star of the show: our DistilBERT implementation                       |
| `bert_model.py`        | BERT encoder (the teacher, not a pre-trained magician by default!)        |
| `modules.py`           | All the building blocks: embeddings, attention, FFN, etc.                |
| `transformer_block.py` | One transformer block to rule them all                                    |
| `loss.py`              | The secret sauce: Knowledge Distillation Loss (KL + CE)                  |
| `config.py`            | Hyperparameters, paths, and all your favorite knobs                       |
| `train.py`             | Training loop, inference, and demo pipeline                              |

---

## 🤖 How Does It Work?

### 1. **Embeddings**

- Token + Position embeddings, added together (because order matters!).
- Dropout for regularization — every transformer’s favorite outfit.

### 2. **Transformer Blocks**

- Each block = MultiHeadSelfAttention → AddNorm → FeedForward → AddNorm.
- Stack them high (BERT) or keep it tight (DistilBERT, with half the layers).

### 3. **Knowledge Distillation**

- The BERT teacher makes predictions.
- The DistilBERT student tries to mimic the teacher’s “soft” outputs (via KL divergence) **and** the hard ground-truth labels (cross-entropy).
- The loss is a weighted sum:  
  `total_loss = α * distillation_loss + (1-α) * student_loss`

### 4. **Training Loop**

- Loads toy data (for fun) or your own dataset (for real results).
- Uses AdamW, batching, and all the modern training tricks.
- Saves the trained student for inference and bragging rights.

---

## 🛠️ Getting Started

### Prereqs

- Python 3.8+
- PyTorch
- transformers (for tokenizer)
- 🤓 Curiosity

### Quickstart

```bash
git clone https://github.com/sahil-gaikwad94/DistilBERT-Paper-implementation.git
cd DistilBERT-Paper-implementation
pip install torch transformers
python train.py
```

### Custom Training

- Swap out the dummy dataset for your own.
- Load pre-trained BERT weights for the teacher in `train.py` for true distillation power.

---

## 💡 Why DistilBERT?

DistilBERT is:
- **40% smaller** than BERT-base
- **60% faster** at inference
- **95%+ of BERT’s performance**

It’s perfect for edge devices, real-time apps, or anyone who likes their NLP fast and frugal.

---

## 📝 Notes & Quirks

- This repo is for education, tinkering, and showing off at NLP meetups.
- For production, use HuggingFace’s `transformers` — unless you like to live on the wild side!
- Teacher model here defaults to random weights for demo. For real distillation, load those BERT weights!

---

## 👨‍🔬 References

- [DistilBERT: A distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

## 🦸‍♂️ Author

- Built by [@sahil-gaikwad94](https://github.com/sahil-gaikwad94) — reach out with questions, PRs, or memes.

---

**Ready to shrink your transformers? Fork, star, and distill some knowledge!**
