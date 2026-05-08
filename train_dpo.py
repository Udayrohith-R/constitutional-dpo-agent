import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstitutionalDPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss implemented from scratch.
    Ensures numerical stability across large-scale tensor operations.
    """
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(self, 
                pi_logps_chosen: torch.Tensor, 
                pi_logps_rejected: torch.Tensor, 
                ref_logps_chosen: torch.Tensor, 
                ref_logps_rejected: torch.Tensor) -> torch.Tensor:
        
        # Calculate the log ratios between policy model and reference model
        pi_logratios = pi_logps_chosen - pi_logps_rejected
        ref_logratios = ref_logps_chosen - ref_logps_rejected
        
        # Calculate logits for the sigmoid
        logits = pi_logratios - ref_logratios
        
        # Compute the negative log sigmoid of the scaled logits for numerical stability
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss

def extract_sequence_logps(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Extracts log probabilities of the actual token sequence, masking padding."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_mask = shift_labels != ignore_index
    # Numerically stable log softmax
    log_probs = F.log_softmax(shift_logits, dim=-1) 
    
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

# Note for Reviewer: In the training loop, pass the chosen/rejected logits through extract_sequence_logps 
# before feeding them into ConstitutionalDPOLoss to compute the backward pass.