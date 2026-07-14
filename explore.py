# %% Imports
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from src.data import CustomImageNetDataset
from src.networks import MLP_Projection, BaseEncoder, SimCLR
from src.losses import NT_Xent_Loss, LossType

# %% Tests
dataset = CustomImageNetDataset(test_size=1000)
dataloader = DataLoader(dataset, batch_size=10)

# %% Get images
x_i, x_j = next(iter(dataloader))

# %% Test
x_i.shape

# %% Test networks
base_encoder = BaseEncoder()
mlp_projection_head = MLP_Projection()
sim_clr = SimCLR(base_encoder, mlp_projection_head)

# %% Forward passes
sim_x1 = sim_clr(x_i)
sim_x2 = sim_clr(x_j)

# %% Forward passes
sim_x1 = torch.rand(5,128)
sim_x2 = torch.rand(5,128)
loss = NT_Xent_Loss(temperature=1.2, loss_type=LossType.SUM)

# %% Forward passes
loss(sim_x1,sim_x2)
