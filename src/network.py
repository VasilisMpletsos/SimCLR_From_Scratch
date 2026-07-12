# %% Imports
from torchvision.models import resnet50
from transformers import ResNetForImageClassification
from torchviz import make_dot
from torch.nn import Module, Linear, ReLU
import torch

class MLP_Projection(Module):
    def __init__(self):
        super().__init__()
        self.hidden = Linear(1000, 524)
        self.head = Linear(524, 128)
        self.sigma = ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigma(x)
        x = self.head(x)
        return x

class BaseEncoder(Module):
    def __init__(self):
        super().__init__()
        self.base_encoder = resnet50(weights=None);

    def forward(self, x):
        x = self.base_encoder(x)
        return x


class SimCLR(Module):

    def __init__(self, base_encoder: BaseEncoder, head: MLP_Projection):
        super().__init__()

        # %% Pytorch Implementation
        self.base_encoder = base_encoder;
        self.projection_head = head;

        # %% HF Implementation
        # base_encoder = ResNetForImageClassification.from_pretrained("microsoft/resnet-50");

    def forward(self, x):
        x = self.base_encoder(x)
        x = self.projection_head(x)
        return x
