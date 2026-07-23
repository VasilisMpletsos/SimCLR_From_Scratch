# %% Imports
import os
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import all_reduce, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import CustomImageNetDataset
from src.losses import LossType, NT_Xent_Loss
from src.networks import BaseEncoder, MLP_Projection, SimCLR

if __name__ == "__main__":
    BATCH_SIZE = 128
    LOG_STEP = 50
    VALIDATION_STEP = 1000
    TRAIN_SIZE = 1_281_167
    NUM_GPUS = 2
    TEST_SIZE = 500

    # DDP Config
    ddp = int(os.getenv("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "Cuda is needed to run multi parallel GPU"
        init_process_group(backend="nccl")
        ddp_rank = int(os.getenv("RANK"))
        ddp_local_rank = int(os.getenv("LOCAL_RANK"))
        ddp_world_size = int(os.getenv("WORLD_SIZE"))
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        raise Exception("Failure to find DDP")

    print(f"Using device {device} for training")

    if master_process:
        writer = SummaryWriter(log_dir="./runs/simclr", comment="First run")

    # %% Set Data
    train_dataset = CustomImageNetDataset(split="train")
    validation_dataset = CustomImageNetDataset(split="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
    )

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Set Model
    base_encoder = BaseEncoder()
    mlp_projection_head = MLP_Projection()
    sim_clr = SimCLR(base_encoder, mlp_projection_head)
    sim_clr.to(device)

    # %% Configure loss & optimizer
    EPOCHS = 10
    nt_xent_loss = NT_Xent_Loss(temperature=1.2, loss_type=LossType.SUM, device=device)
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
            x_i = x_i.to(device, non_blocking=True)
            x_j = x_j.to(device, non_blocking=True)
            # Forward passes
            sim_x1 = sim_clr(x_i)
            sim_x2 = sim_clr(x_j)
            loss = nt_xent_loss(sim_x1, sim_x2)
            global_step = (epoch * TRAIN_SIZE) + i + 1
            if (i % LOG_STEP == 0) and master_process:
                step_loss = loss.cpu().item()
                writer.add_scalar("Loss/train", step_loss, global_step)
                print(f"Global Step {global_step} | Loss:{step_loss}")
            loss.backward()
            all_reduce(loss, op=dist.ReduceOp.AVG)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i > 0 and i % VALIDATION_STEP == 0:
                print("GOING INTO VALIDATION MODE")
                sim_clr.eval()
                with torch.no_grad():
                    total_loss = []
                    for j, (x_i, x_j) in enumerate(validation_dataloader):
                        x_i = x_i.to(device, non_blocking=True)
                        x_j = x_j.to(device, non_blocking=True)
                        sim_x1 = sim_clr(x_i)
                        sim_x2 = sim_clr(x_j)
                        loss = nt_xent_loss(sim_x1, sim_x2)
                        all_reduce(loss, op=dist.ReduceOp.AVG)
                        total_loss.append(loss.cpu().item())
                    total_loss = np.asarray(total_loss).mean()
                    if master_process:
                        writer.add_scalar(
                            "Loss/validation",
                            total_loss.item(),
                            global_step,
                        )
                        if total_loss < best_loss:
                            best_loss = total_loss
                            torch.save(
                                base_encoder.state_dict(),
                                f"./models/base_{global_step}",
                            )

    if master_process:
        writer.close()
        destroy_process_group()
