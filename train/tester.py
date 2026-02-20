import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import evaluate_predictions
from utils.logger import log_results, update_results_csv
from utils.check_point import save_checkpoint
from data.dataset import prepare_inputs
from train.losses import compute_loss
import torch.nn as nn
from pathlib import Path
from utils.timing import timeit
from train.trainer import move_to_device, GPUTransform

@timeit
def test_model(cfg, net, allsetDataloader, run_wandb, log_file_path):
    
    print('Performing Testing')
    # Clear GPU memory
    gpu_transform = GPUTransform().to(cfg.system.device)
    torch.cuda.empty_cache()

    best_paths = get_best_weights(log_file_path)
    
    phase = "test"
    epoch = 0
    
    j = 0
    is_train = False
    for load_file in best_paths:
    
        net.load_state_dict(torch.load(load_file, map_location=torch.device(cfg.system.device), weights_only=True))
        net.eval()  #switches certain layers into inference mode e.g., BatchNorm, Dropout
        
        test_preds_np = []
        test_trues_np = []
        
        for i, data in enumerate(tqdm(allsetDataloader['test'], desc="Testing...")):
            inputs, targets = prepare_inputs(data, cfg)
            
            targets = move_to_device(targets, cfg.system.device)
            inputs = move_to_device(inputs, cfg.system.device)
            if cfg.data.input_feature_type != "explicit_feature":
                inputs["images"] = gpu_transform(inputs["images"], is_train)
            
            with torch.no_grad():
                preds = forward_pass(net, inputs)
            targets = targets.float()
            
            y_true = targets.detach().cpu().numpy()
    
            y_pred = preds.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
    
            test_preds_np.extend(y_true)
            test_trues_np.extend(y_pred)
            
        metrics, cm, classification_report = evaluate_predictions(
            test_trues_np,
            test_preds_np,
            phase,
            cfg.model.classes_name
        )
    
        log_results(
            epoch=epoch,
            metrics=metrics,
            run_wandb=run_wandb,
            cm=cm,
            classification_report=classification_report,
            phase=phase,
            log_file_path=log_file_path,
            classes=cfg.model.classes_name
        )
        
        update_results_csv(cfg, metrics)

@timeit
def forward_pass(net, inputs):
    """Run forward pass."""
    return net(inputs)

@timeit
def get_best_weights(directory):
    directory = Path(directory)
    return list(directory.glob("*best.tar"))
