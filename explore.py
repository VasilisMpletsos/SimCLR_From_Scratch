# %% Imports
import torch
import torch.nn.functional as F

# %% Test
x1 = torch.rand(5, 128)
x2 = torch.rand(5, 128)
temperature = torch.tensor(1.2)

# %% Test
x1 = F.normalize(x1)
x2 = F.normalize(x2)

# %% Test
logits = (x1 @ x2.T) / temperature

# %% Test
logits

# %% Test
positives = -F.logsigmoid(logits.diag())
# %% Test
positives
# %% Test
negatives = logits[neg_mask.bool()]
# %% Test
negatives = -F.logsigmoid(-negatives)
# %% Test
negatives
