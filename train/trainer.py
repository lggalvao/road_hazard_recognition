import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import evaluate_predictions
from utils.logger import log_results
from utils.check_point import save_checkpoint
from data.dataset import prepare_inputs
from train.losses import compute_loss
import torch.nn as nn
from utils.timing import timeit, print_average_timings
import time
import kornia.augmentation as K
import torch.profiler
import logging
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

logger = logging.getLogger("hazard_recognition")


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
        )

        self.normalize = K.Normalize(
            mean=torch.tensor([0.485,0.456,0.406]),
            std=torch.tensor([0.229,0.224,0.225])
        )

    def forward(self, x, is_train=True):

        B, T, C, H, W = x.shape
        x = x.float() / 255.0
        x = x.view(B * T, C, H, W)

        if is_train:
            x = self.augs(x)

        x = self.normalize(x)

        x = x.view(B, T, C, H, W)
        return x



# -------------------------
# Main Training Loop
# -------------------------
@timeit
def train_model(cfg, net, allsetDataloader, optimizer, exp_lr_scheduler, criterion, early_stopping, run_wandb, log_file_path):

    scaler = GradScaler(enabled=cfg.training.amp_enabled)
    logger.info(f"AMP enabled: {scaler.is_enabled()}")
    gpu_transform = GPUTransform().to(cfg.system.device)
    previous_train_F1 = None
    best_val_f1 = 0
    f1_train_val_gap = 0
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs-1}")
 
        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            
            print(cfg.training.run_epoch_profile)
            if epoch == 0 and cfg.training.run_epoch_profile:
                # Function to inspect CPU and GPU usage
                logger.info("Running Torch Profile")
                run_epoch_profile(
                    net,
                    allsetDataloader[phase],
                    optimizer,
                    criterion,
                    cfg,
                    is_train,
                    gpu_transform,
                    exp_lr_scheduler,
                    scaler
                )
                logger.info("Torch Profile Done")
            
            avg_loss, targets, preds = run_epoch(
                net,
                allsetDataloader[phase],
                optimizer,
                criterion,
                cfg,
                is_train,
                gpu_transform,
                exp_lr_scheduler,
                scaler
            )

            metrics = evaluate_predictions(targets, preds, phase, cfg.model.classes_name)
            metrics["loss"] = avg_loss
            metrics["best_val_f1"] = best_val_f1

            log_results(
                epoch = epoch,
                metrics = metrics,
                run_wandb = run_wandb,
                phase = phase,
                log_file_path = log_file_path
            )

            if is_train:
                if cfg.training.lr_scheduler == "StepLR":
                    exp_lr_scheduler.step()
                previous_train_F1 = metrics["f1_macro"]

            else:  # validation
                current_val_f1_macro = metrics["f1_macro"]
                f1_train_val_gap = previous_train_F1 - current_val_f1_macro
                
                early_stopping(
                    avg_loss,
                    net
                )
                
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    print_average_timings()
                    return
                
                if current_val_f1_macro > best_val_f1 and f1_train_val_gap < 0.15:
                    print(f"Val F1 Macro increased: {best_val_f1:3f}-->{current_val_f1_macro:.3f}")
                    best_val_f1 = current_val_f1_macro
                    save_checkpoint(net, log_file_path, "f1_macro_val_best")

        print_average_timings()
        
        if f1_train_val_gap > 0.2:
            logger.info("Training stopped due to big difference between tain and val F1 macro.")
            break 

@timeit
def run_epoch(net, dataloader, optimizer, criterion, cfg, is_train, gpu_transform, exp_lr_scheduler, scaler):

    net.train() if is_train else net.eval()
    
    epoch_loss = 0.0
    move_to_device_time = 0
    prepare_inputs_time = 0
    gpu_transform_time = 0
    forward_pass_time = 0
    compute_loss_time = 0
    backward_time = 0

    epoch_preds, epoch_targets = [], []

    for data in tqdm(dataloader, desc="Training..." if is_train else "Validating..."):
   
        t1 = time.time()
        #inputs, targets = prepare_inputs(data, cfg)
        inputs, targets = data
        t2 =time.time()
        
        prepare_inputs_time += (t2 - t1)

        t1 = time.time()
        targets = targets.to(cfg.system.device, non_blocking=True)
        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = [t.to(cfg.system.device, non_blocking=True) for t in v]
            else:
                inputs[k] = v.to(cfg.system.device, non_blocking=True)
        #targets = move_to_device(targets, cfg.system.device)
        #inputs = move_to_device(inputs, cfg.system.device)
        t2 =time.time()
        move_to_device_time += (t2 - t1)
        
        
        for k, v in inputs.items():
            if isinstance(v, list):
                print(k, len(v))
            else:
                print(k, v.shape)
        
        if cfg.data.input_feature_type != "explicit_feature":
            t1 = time.time()
            inputs["images"] = gpu_transform(inputs["images"], is_train)
            t2 =time.time()
            gpu_transform_time += (t2 - t1)

        if is_train:
            #optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            
            if cfg.training.amp_enabled:

                with autocast(enabled=cfg.training.amp_enabled):
                    t1 = time.time()
                    preds = forward_pass(net, inputs)
                    t2 =time.time()
                    forward_pass_time += (t2 - t1)
                    
                    t1 = time.time()
                    loss = compute_loss(criterion, preds, targets, cfg)
                    t2 =time.time()
                    compute_loss_time += (t2 - t1)
            
            else:
                t1 = time.time()
                preds = forward_pass(net, inputs)
                t2 =time.time()
                forward_pass_time += (t2 - t1)
                
                t1 = time.time()
                loss = compute_loss(criterion, preds, targets, cfg)
                t2 =time.time()
                compute_loss_time += (t2 - t1)

            if is_train:
                if cfg.training.amp_enabled:
                    scaler.scale(loss).backward()
                    # Clipping should be done on unscaled gradients.
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    t1 = time.time()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training.clip_grad)
                    optimizer.step()
                    t2 =time.time()
                    backward_time += (t2 - t1)
                
                if cfg.training.lr_scheduler == "CosineAnnealingLR" or cfg.training.lr_scheduler == "CosineAnnealingLRWarmUp":
                    exp_lr_scheduler.step()
            
            #if is_train and torch.rand(1).item() < 0.001:
            #    print("LR:", optimizer.param_groups[0]["lr"])

        epoch_loss += loss.item()

        # ---- Metrics ----
        epoch_targets.extend(targets.detach().cpu().numpy())
        epoch_preds.extend(torch.argmax(preds, dim=1).detach().cpu().numpy())

    print(f"prepare_inputs_time: {prepare_inputs_time:.3f}")
    print(f"move_to_device_time: {move_to_device_time:.3f}")
    print(f"gpu_transform_time: {gpu_transform_time:.3f}")
    print(f"forward_pass_time: {forward_pass_time:.3f}")
    print(f"compute_loss_time: {compute_loss_time:.3f}")
    print(f"backward_time: {backward_time:.3f}")
    
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss, epoch_targets, epoch_preds


def run_epoch_profile(net, dataloader, optimizer, criterion, cfg, is_train, gpu_transform, exp_lr_scheduler, scaler):

    net.train() if is_train else net.eval()

    max_steps = 100

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=10,
            active=90,
            repeat=1
        ),
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        for step, data in enumerate(dataloader):

            #inputs, targets = prepare_inputs(data, cfg)
            #targets = move_to_device(targets, cfg.system.device)
            #inputs = move_to_device(inputs, cfg.system.device)
            
            inputs, targets = data
            targets = targets.to(cfg.system.device, non_blocking=True)
            for k, v in inputs.items():
                if isinstance(v, list):
                    inputs[k] = [t.to(cfg.system.device, non_blocking=True) for t in v]
                else:
                    inputs[k] = v.to(cfg.system.device, non_blocking=True)

            if cfg.data.input_feature_type != "explicit_feature":
                inputs["images"] = gpu_transform(inputs["images"], is_train)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                preds = forward_pass(net, inputs)
                loss = compute_loss(criterion, preds, targets, cfg)

                if is_train:
                    loss.backward()
                    optimizer.step()

            prof.step()

            if step >= max_steps - 1:
                break

    print(prof.key_averages().table(sort_by="cuda_time_total"))



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


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch