import torch


def get_optimizer(cfg, net):
    """
    Returns a PyTorch optimizer for staged training.
    
    Features:
    - Stage 0: train the entire network (no freezing)
    - Stage 1: train LSTM + FC only
    - Stage 2: fine-tune CNN layer4 + LSTM + FC
    - Automatically detects trainable parameters
    - Supports Adam or SGD
    - Supports per-parameter learning rates
    """

    # ------------------------
    # Determine parameter groups
    # ------------------------
    if cfg.training.stage == 0:
        if cfg.training.global_lr:
            param_groups = [{'params': [p for p in net.parameters() if p.requires_grad],
                            'lr': cfg.training.backbone_lr}]
        else:
            backbone_decay = []
            backbone_no_decay = []
            head_params = []
            
            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                if "classifier" in name:  # adjust depending on your head naming
                    head_params.append(param)
                elif param.ndim == 1 or name.endswith(".bias"):
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)
            
            param_groups = [
                {"params": backbone_decay, "lr": cfg.training.backbone_lr, "weight_decay": cfg.training.weight_decay},
                {"params": backbone_no_decay, "lr": cfg.training.backbone_lr, "weight_decay": 0.0},
                {"params": head_params, "lr": cfg.training.head_lr, "weight_decay": cfg.training.weight_decay},
            ]
        
        stage_name = "Stage 0: train entire network"

    elif cfg.training.stage == 1:
        # Stage 1: train only LSTM + FC
        param_groups = [{'params': [p for p in net.lstm.parameters() if p.requires_grad],
                         'lr': cfg.training.backbone_lr},
                        {'params': [p for p in net.fc.parameters() if p.requires_grad],
                         'lr': cfg.training.backbone_lr}]
        stage_name = "Stage 1: train LSTM + FC"

    elif cfg.training.stage == 2:
        # Stage 2: fine-tune CNN layer4 + LSTM + FC
        param_groups = [
            {'params': [p for p in net.resnet[-1].parameters() if p.requires_grad],
             'lr': cfg.training.cnn_lr},
            {'params': [p for p in net.lstm.parameters() if p.requires_grad],
             'lr': cfg.training.backbone_lr},
            {'params': [p for p in net.fc.parameters() if p.requires_grad],
             'lr': cfg.training.backbone_lr},
        ]
        stage_name = "Stage 2: fine-tune CNN layer4 + LSTM + FC"

    else:
        raise ValueError(f"Unsupported training stage: {cfg.training.stage}")

    # ------------------------
    # Sanity check: trainable params
    # ------------------------
    total_params = sum(p.numel() for group in param_groups for p in group['params'])
    print(f"{stage_name} | Trainable parameters: {total_params:,}")

    # ------------------------
    # Select optimizer
    # ------------------------
    optimizer_type = cfg.training.optimizer
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            param_groups,
            #weight_decay=cfg.training.weight_decay
        )
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=getattr(cfg.training, 'momentum', 0.9),
            #weight_decay=cfg.training.weight_decay
        )
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            param_groups,
            #weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    return optimizer
