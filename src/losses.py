import torch
from torch.nn import Module
from torch.linalg import vector_norm
from enum import Enum
from torch.functional import F

class LossType(Enum):
    MEAN = 0
    SUM = 1


class NT_Xent_Loss(Module):

    def __init__(self, temperature: float = 1.1, loss_type: LossType = LossType.MEAN):
        super().__init__()
        self.temp = temperature
        self.loss_type = loss_type

    def forward(self,z1, z2):
        N = z1.shape[0]
        double_N = 2 * N
        z = torch.cat([z1,z2], dim=0)
        norm_z = F.normalize(z)
        similarities = z @ z.T
        exp_similarities = torch.exp(similarities/self.temp)
        masked_exp_similarities = exp_similarities.masked_fill(torch.eye(double_N, double_N, dtype=torch.bool), torch.tensor(0))
        sums = masked_exp_similarities.sum(dim=1, keepdim=True)
        losses = exp_similarities/sums
        valid_ids = torch.cat([torch.eye(N),torch.eye(N)], dim=0)
        valid_ids = valid_ids @ valid_ids.T
        valid_mask = valid_ids.bool()
        valid_mask.fill_diagonal_(False)
        positive_losses = losses[valid_ids.bool()]

        if (self.loss_type == LossType.MEAN):
            return positive_losses.mean()
        else:
            return positive_losses.sum()
