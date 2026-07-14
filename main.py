# %% Imports
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import CustomImageNetDataset
from src.losses import LossType, NT_Xent_Loss
from src.networks import BaseEncoder, MLP_Projection, SimCLR

if __name__ == "__main__":
    # %% Set Data
    dataset = CustomImageNetDataset(test_size=1000)
    dataloader = DataLoader(
        dataset, batch_size=64, num_workers=4, pin_memory=True, prefetch_factor=2
    )

    # Set Model
    base_encoder = BaseEncoder()
    mlp_projection_head = MLP_Projection()
    sim_clr = SimCLR(base_encoder, mlp_projection_head)
    sim_clr.to("cuda")

    # %% Configure loss & optimizer
    EPOCHS = 10
    nt_xent_loss = NT_Xent_Loss(temperature=1.2, loss_type=LossType.SUM, device="cuda")
    optimizer = AdamW(params=sim_clr.parameters(), lr=1e-2)

    # %% Training
    for epoch in range(EPOCHS):
        for i, (x_i, x_j) in enumerate(dataloader):
            x_i = x_i.to("cuda", non_blocking=True)
            x_j = x_j.to("cuda", non_blocking=True)
            # Forward passes
            sim_x1 = sim_clr(x_i)
            sim_x2 = sim_clr(x_j)
            loss = nt_xent_loss(sim_x1, sim_x2)
            # if i % 10 == 0:
            step_loss = loss.cpu().item()
            print(f"Epoch {epoch} | Step {i} | Loss:{step_loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
