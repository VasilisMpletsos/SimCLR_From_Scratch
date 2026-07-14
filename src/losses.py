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
        # Cache for batch-size-dependent tensors
        self._cached_N = None
        self._cached_eye_mask = None
        self._cached_valid_mask = None

    def _get_masks(self, N):
        """Cache masks for a given batch size to avoid recreating them."""
        if self._cached_N != N:
            self._cached_N = N
            double_N = 2 * N

            # Cache the eye mask for diagonal
            self._cached_eye_mask = torch.eye(
                double_N, double_N, dtype=torch.bool, device=self.device
            )

            # Cache the valid positive pairs mask
            eye_N = torch.eye(N, device=self.device)
            valid_ids = torch.cat([eye_N, eye_N], dim=0)
            valid_ids = valid_ids @ valid_ids.T
            self._cached_valid_mask = valid_ids.bool()
            self._cached_valid_mask.fill_diagonal_(False)

        return self._cached_eye_mask, self._cached_valid_mask

    def forward(self, z1, z2):
        N = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        norm_z = F.normalize(z)
        similarities = norm_z @ norm_z.T
        exp_similarities = torch.exp(similarities / self.temp)

        # Get cached masks
        eye_mask, valid_mask = self._get_masks(N)

        # Mask out diagonal
        # Type assertion to satisfy type checker
        assert eye_mask is not None and valid_mask is not None
        masked_exp_similarities = exp_similarities.masked_fill(eye_mask, 0.0)
        sums = masked_exp_similarities.sum(dim=1, keepdim=True)
        losses = exp_similarities / sums

        # Extract positive pairs
        positive_losses = losses[valid_mask]

        if self.loss_type == LossType.MEAN:
            return positive_losses.mean()
        else:
            return positive_losses.sum()
