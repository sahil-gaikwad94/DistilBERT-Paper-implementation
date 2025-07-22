import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha_distil=0.5, alpha_hard=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_distil = alpha_distil
        self.alpha_hard = alpha_hard

        self.cross_entropy_loss = nn.CrossEntropyLoss() # For hard targets
        self.kl_divergence_loss = nn.KLDivLoss(reduction='batchmean') # For soft targets

    def forward(self, student_logits, teacher_logits, labels):
        # 1. Calculate Distillation Loss (KL Divergence)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        distillation_loss = self.kl_divergence_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)

        # 2. Calculate Student Loss (Cross-Entropy with hard targets)
        student_hard_loss = self.cross_entropy_loss(student_logits, labels)

        # 3. Combine losses with weighting factors
        total_loss = (self.alpha_distil * distillation_loss) + (self.alpha_hard * student_hard_loss)

        return total_loss


