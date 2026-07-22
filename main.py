# %% Imports
import os
from datetime import datetime

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import CustomImageNetDataset
from src.losses import LossType, NT_Xent_Loss
from src.networks import BaseEncoder, MLP_Projection, SimCLR

if __name__ == "__main__":
    writer = SummaryWriter(log_dir="./runs/simclr", comment="First run")
    BATCH_SIZE = 128
    LOG_STEP = 50
    VALIDATION_STEP = 1000
    TRAIN_SIZE = 1_281_167

    # %% Set Data
    train_dataset = CustomImageNetDataset(split="train")
    validation_dataset = CustomImageNetDataset(split="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
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
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(TRAIN_SIZE // BATCH_SIZE) * 3,  # Maximum number of iterations.
        eta_min=1e-4,
    )  # Minimum learning rate.

    best_loss = np.inf

    # %% Training
    for epoch in range(EPOCHS):
        for i, (x_i, x_j) in enumerate(train_dataloader):
            sim_clr.train()
            x_i = x_i.to("cuda", non_blocking=True)
            x_j = x_j.to("cuda", non_blocking=True)
            # Forward passes
            sim_x1 = sim_clr(x_i)
            sim_x2 = sim_clr(x_j)
            loss = nt_xent_loss(sim_x1, sim_x2)
            if i % LOG_STEP == 0:
                step_loss = loss.cpu().item()
                writer.add_scalar("Loss/train", step_loss, (epoch * TRAIN_SIZE) + i + 1)
                print(f"Epoch {epoch} | Step {i} | Loss:{step_loss}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i > 0 and i % VALIDATION_STEP == 0:
                print("GOING INTO VALIDATION MODE")
                sim_clr.eval()
                with torch.no_grad():
                    total_loss = []
                    for j, (x_i, x_j) in enumerate(validation_dataloader):
                        x_i = x_i.to("cuda", non_blocking=True)
                        x_j = x_j.to("cuda", non_blocking=True)
                        sim_x1 = sim_clr(x_i)
                        sim_x2 = sim_clr(x_j)
                        loss = nt_xent_loss(sim_x1, sim_x2)
                        total_loss.append(loss.cpu().item())
                    total_loss = np.asarray(total_loss).mean()
                    print(
                        f"Epoch {epoch} | Step {i} | Validaiton Loss:{total_loss.item()}"
                    )
                    writer.add_scalar(
                        "Loss/validation",
                        total_loss.item(),
                        (epoch * TRAIN_SIZE) + i + 1,
                    )
                    if total_loss < best_loss:
                        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                        torch.save(base_encoder.state_dict(), f"./models/base_{now}")
