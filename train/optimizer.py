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
        # Stage 0: train everything
        param_groups = [{'params': [p for p in net.parameters() if p.requires_grad],
                         'lr': cfg.training.learning_rate}]
        stage_name = "Stage 0: train entire network"

    elif cfg.training.stage == 1:
        # Stage 1: train only LSTM + FC
        param_groups = [{'params': [p for p in net.lstm.parameters() if p.requires_grad],
                         'lr': cfg.training.learning_rate},
                        {'params': [p for p in net.fc.parameters() if p.requires_grad],
                         'lr': cfg.training.learning_rate}]
        stage_name = "Stage 1: train LSTM + FC"

    elif cfg.training.stage == 2:
        # Stage 2: fine-tune CNN layer4 + LSTM + FC
        param_groups = [
            {'params': [p for p in net.resnet[-1].parameters() if p.requires_grad],
             'lr': cfg.training.cnn_lr},
            {'params': [p for p in net.lstm.parameters() if p.requires_grad],
             'lr': cfg.training.learning_rate},
            {'params': [p for p in net.fc.parameters() if p.requires_grad],
             'lr': cfg.training.learning_rate},
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
    optimizer_type = cfg.training.optimizer.lower()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=cfg.training.weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=getattr(cfg.training, 'momentum', 0.9),
            weight_decay=cfg.training.weight_decay
        )
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    return optimizer


#def get_optimizer(cfg, net):
#                
#    if cfg.training.optimizer == 'Adam':
#        return torch.optim.Adam(net.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
#    
#    elif cfg.training.optimizer == 'SGD':
#        return torch.optim.SGD(net.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay, momentum=cfg.training.momentum)