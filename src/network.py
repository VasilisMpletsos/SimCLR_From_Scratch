# %% Imports
from torchvision.models import resnet50
from transformers import ResNetForImageClassification
from torchviz import make_dot
import torch
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# # %% Inspect Pytorch Network
# base_network = resnet50();
# base_network.__class__

# %% Inspect HF Network
base_network = ResNetForImageClassification.from_pretrained("microsoft/resnet-50");
