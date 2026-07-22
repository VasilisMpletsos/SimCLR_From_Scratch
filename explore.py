# %% Imports
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.functional import F
from torch.nn import Module
from torch.utils.data import DataLoader

from src.data import CustomImageNetDataset
from src.losses import LossType, NT_Xent_Loss
from src.networks import BaseEncoder, MLP_Projection, SimCLR

# %% Tests
eye_mask = torch.eye(5, 5)
valid_ids = torch.cat([eye_mask, eye_mask], dim=0)
valid_ids = valid_ids @ valid_ids.T
valid_mask = valid_ids.bool().fill_diagonal_(False)
# %% Tests
dataset = CustomImageNetDataset(test_size=1000)
dataloader = DataLoader(dataset, batch_size=10)

# %% Get images
x_i, x_j = next(iter(dataloader))

# %% Test networks
base_encoder = BaseEncoder()
mlp_projection_head = MLP_Projection()
sim_clr = SimCLR(base_encoder, mlp_projection_head)

# %% Forward passes
sim_x1 = sim_clr(x_i)
sim_x2 = sim_clr(x_j)
loss = NT_Xent_Loss(temperature=1.2, loss_type=LossType.MEAN)
