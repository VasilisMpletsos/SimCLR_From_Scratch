# %% Imports
from src.augmentation import image_augmentation
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from src.network import MLP_Projection, BaseEncoder, SimCLR

# %% Tests
ds = load_dataset("evanarlian/imagenet_1k_resized_256", streaming=True, split="train").shuffle(seed=42, buffer_size=1)

# %% Get images
image = next(iter(ds))['image']

# %% Tensor
tensor_image_x1 = image_augmentation(image)
tensor_image_x2 = image_augmentation(image)

# %% Test networks
base_encoder = BaseEncoder()
mlp_projection_head = MLP_Projection()
sim_clr = SimCLR(base_encoder, mlp_projection_head)

# %% Forward passes
sim_x1 = sim_clr(tensor_image_x1.view(1,3,224,224))
sim_x2 = sim_clr(tensor_image_x2.view(1,3,224,224))
# %% Forward passes
(sim_x1 - sim_x2).sum()
