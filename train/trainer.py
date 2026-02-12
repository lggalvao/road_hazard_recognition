import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import evaluate_predictions
from utils.logger import log_results, update_results_csv
from utils.check_point import save_checkpoint
from data.dataset import prepare_inputs
from train.losses import compute_loss
import torch.nn as nn
from utils.timing import timeit, print_average_timings
import time
import kornia.augmentation as K

class GPUTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.augs = torch.nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.02,
                p=0.8
            ),
            K.RandomGaussianBlur(
                kernel_size=(3,3),
                sigma=(0.1,1.0),
                p=0.3
            ),
        )

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485,0.456,0.406]),
            std=torch.tensor([0.229,0.224,0.225])
        )
        
    def forward(self, x):
        # x: (B, T, C, H, W)
    
        B, T, C, H, W = x.shape
    
        x = x.float() / 255.0
    
        output = []
    
        for b in range(B):
            seq = x[b]                 # (T, C, H, W)
            seq = self.augs(seq)       # Kornia sees this as batch=T
            output.append(seq)
    
        x = torch.stack(output)        # (B, T, C, H, W)
    
        x = self.normalize(
            x.view(B*T, C, H, W)
        ).view(B, T, C, H, W)
    
        return x



# -------------------------
# Main Training Loop
# -------------------------
@timeit
def train_model(cfg, net, allsetDataloader, optimizer, exp_lr_scheduler, criterion, early_stopping, run_wandb, log_file_path):

    gpu_transform = GPUTransform().to(cfg.system.device)
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs-1}")
 
        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            
            avg_loss, targets, preds = run_epoch(
                net,
                allsetDataloader[phase],
                optimizer,
                criterion,
                cfg,
                is_train,
                gpu_transform
            )

            metrics = evaluate_predictions(targets, preds, phase, cfg.model.classes_name)
            metrics["loss"] = avg_loss

            log_results(
                epoch = epoch,
                metrics = metrics,
                run_wandb = run_wandb,
                phase = phase,
                log_file_path = log_file_path
            )

            if is_train:
                exp_lr_scheduler.step()

            else:  # validation
                f1_macro = metrics["f1_macro"]
                early_stopping(
                    #-f1_macro,
                    avg_loss,
                    net
                )
                #save_checkpoint(net, log_file_path, f"epoch{epoch}")
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    print_average_timings()
                    return
        
        print_average_timings()

@timeit
def run_epoch(net, dataloader, optimizer, criterion, cfg, is_train, gpu_transform):

    net.train() if is_train else net.eval()

    cpu_dataloader_time = 0.0
    gpu_transform_time = 0.0
    gpu_time = 0.0
    epoch_loss = 0.0

    epoch_preds, epoch_targets = [], []

    end = time.time()

    for data in tqdm(dataloader, desc="Training..." if is_train else "Validating..."):

        # ---- DataLoader wait time ----
        start = time.time()
        cpu_dataloader_time += (start - end)

        inputs, targets = prepare_inputs(data, cfg)
        
        inputs = move_to_device(inputs, "cuda")
        t1 = time.time()
        inputs["images"] = gpu_transform(inputs["images"])
        t2 = time.time()
        gpu_transform_time += (t2 - t1)

        if is_train:
            optimizer.zero_grad()

        # ---- GPU compute time ----
        torch.cuda.synchronize()
        t1 = time.time()

        with torch.set_grad_enabled(is_train):
            preds = forward_pass(net, inputs)
            #targets = torch.argmax(targets.float(), dim=1)
            loss = compute_loss(criterion, preds, targets, cfg)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training.clip_grad)
                optimizer.step()

        torch.cuda.synchronize()
        t2 = time.time()
        gpu_time += (t2 - t1)

        epoch_loss += loss.item()

        # ---- Metrics ----
        epoch_targets.extend(targets.detach().cpu().numpy())
        epoch_preds.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

        end = time.time()

    print(f"\nEpoch DataLoader wait time: {cpu_dataloader_time:.2f}s")
    print(f"Epoch GPU compute time: {gpu_transform_time:.2f}s")
    print(f"Epoch GPU compute time: {gpu_time:.2f}s")

    avg_loss = epoch_loss / len(dataloader)
    return avg_loss, epoch_targets, epoch_preds


@timeit
def forward_pass(net, inputs):
    """Run forward pass."""
    return net(inputs)


def estimate_training_time(model, dataset, batch_size, epochs, time_per_epoch_sec):
    time_per_step_sec = time_per_epoch_sec/(len(dataset["train"])* batch_size)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    steps_per_epoch = len(dataset["train"]) * batch_size
    total_steps = steps_per_epoch * epochs
    estimated_time_min = total_steps * time_per_step_sec / 60
    estimated_time_hours = total_steps * time_per_step_sec / 3600
    
    return {
        "trainable_params": total_params,
        "dataset_samples": len(dataset["train"]),
        "total_steps": total_steps,
        "estimated_time_min": estimated_time_min,
        "estimated_time_hours": estimated_time_hours
    }


def estimate_training_time_flops(model, batch_size, gpu_type="A100", safety_factor=1.4):
    """
    Rough estimate of training time per step using FLOPs / GPU throughput method.

    Args:
        model (torch.nn.Module): PyTorch model
        batch_size (int): Batch size for training
        gpu_type (str): One of ["A100", "H100", "RTX4090"]
        safety_factor (float): Factor to account for memory/I/O overhead (default 1.4)

    Returns:
        dict: Estimated FLOPs per step and time per step in seconds
    """
    # 1. Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Estimate FLOPs per training step (forward + backward)
    # Rough heuristic: 6 FLOPs per parameter per sample
    flops_per_step = 6 * num_params * batch_size

    # 3. GPU theoretical throughput (FLOPs/sec)
    gpu_throughputs = {
        "A100": 312e12,     # FP16 TFLOPs
        "H100": 1000e12,    # FP8 TFLOPs
        "RTX4090": 82e12,   # FP16 TFLOPs
        "RTX2080": 5.99,
    }

    if gpu_type not in gpu_throughputs:
        raise ValueError(f"Unknown GPU type: {gpu_type}")

    gpu_flops_per_sec = gpu_throughputs[gpu_type]

    # 4. Time per step in seconds
    time_per_step = (flops_per_step / gpu_flops_per_sec) * safety_factor

    return {
        "trainable_params": num_params,
        "flops_per_step": flops_per_step,
        "estimated_time_per_step_sec": time_per_step
    }


import torch
import kornia.augmentation as K

def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch