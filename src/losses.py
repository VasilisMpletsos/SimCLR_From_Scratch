from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn import Module


class LossType(Enum):
    MEAN = 0
    SUM = 1


class NT_Xent_Loss(Module):
    def __init__(
        self,
        temperature: float = 1.1,
        loss_type: LossType = LossType.MEAN,
        device="cpu",
    ):
        super().__init__()
        self.temp = temperature
        self.loss_type = loss_type
        self.device = device

    def forward(self, z1, z2):
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        norm_z = F.normalize(z)
        similarities = norm_z @ norm_z.T
        exp_similarities = torch.exp(similarities / self.temp)

        eye_mask = torch.eye(N, N, device=self.device)
        valid_ids = torch.cat([eye_mask, eye_mask], dim=0)
        valid_ids = valid_ids @ valid_ids.T
        valid_mask = valid_ids.bool().fill_diagonal_(False)

        masked_exp_similarities = exp_similarities.masked_fill(valid_mask, 0.0)
        sums = masked_exp_similarities.sum(dim=1, keepdim=True)
        losses = -torch.log(exp_similarities / sums)

        # Extract positive pairs
        positive_losses = losses[valid_mask]

        if self.loss_type == LossType.MEAN:
            return positive_losses.mean()
        else:
            return positive_losses.sum()


class NT_Logistic_Loss(Module):
    def __init__(
        self,
        temperature: float = 1.1,
        loss_type: LossType = LossType.MEAN,
        device="cpu",
    ):
        super().__init__()
        self.temp = temperature
        self.loss_type = loss_type
        self.device = device

    def forward(self, z1, z2):

        N = z1.shape[0]

        u = F.normalize(z1, dim=-1)
        v = F.normalize(z2, dim=-1)

        logits = (u @ v.T) / self.temp

        neg_mask = torch.ones(N, N, device=self.device) - torch.eye(
            N, N, device=self.device
        )

        positives = -F.logsigmoid(logits.diag())
        negatives = logits[neg_mask.bool()]
        negatives = -F.logsigmoid(-negatives)

        losses = torch.cat([positives, negatives])

        if self.loss_type == LossType.MEAN:
            return losses.mean()
        else:
            return losses.sum()
