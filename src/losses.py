import torch
from torch.nn import Module
from torch.linalg import vector_norm
from enum import Enum

class LossType(Enum):
    MEAN = 0
    SUM = 1


class NT_Xent_Loss(Module):

    def __init__(self, temperature: float = 1.1, loss_type: LossType = LossType.MEAN):
        super().__init__()
        self.temp = temperature
        self.K_NOT_I_MASK = torch.ones(5,5) - torch.eye(5, 5)
        self.loss_type = loss_type

    def forward(self,z1, z2):
        norm_z1 = vector_norm(z1, dim=1, keepdim=True)
        norm_z2 = vector_norm(z2, dim=1, keepdim=True)

        # i,j view
        sim_calcs = (z1 @ z2.T)/(norm_z1 * norm_z2.T)
        exp_calc = torch.exp(sim_calcs/self.temp)
        exp_sums = (exp_calc * self.K_NOT_I_MASK).sum(dim=1, keepdim=True)
        losses = -torch.log(exp_calc/exp_sums)
        view_ij_losses = torch.diag(losses)

        # j,i view
        # I am doing also j,i because as stated in the paper "The final loss is computed across all positive pairs, both (i, j) and (j, i), in a mini-batch."
        sim_calcs = (z2 @ z1.T)/(norm_z2 * norm_z1.T)
        exp_calc = torch.exp(sim_calcs/self.temp)
        exp_sums = (exp_calc * self.K_NOT_I_MASK).sum(dim=1, keepdim=True)
        losses = -torch.log(exp_calc/exp_sums)
        view_ji_losses = torch.diag(losses)

        # Total loss
        losses = torch.stack([view_ij_losses,view_ji_losses])

        if (self.loss_type == LossType.MEAN):
            return losses.mean()
        else:
            return losses.sum()
