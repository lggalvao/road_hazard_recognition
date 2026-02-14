import torch
from utils.config import Config
import json
from dataclasses import asdict, is_dataclass
import numpy as np
from utils.timing import timeit


@timeit
def save_checkpoint(net, path, name):
    torch.save(net.state_dict(), f"{path}/{name}.tar")

def make_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # convert torch.Tensor → list
    elif isinstance(obj, torch.device):
        return str(obj)      # convert torch.device → "cuda:0"
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # convert numpy.ndarray → list
    elif is_dataclass(obj):
        return {k: make_json_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj

def save_config(cfg: Config, path: str):
    """Save nested dataclass config to JSON safely."""
    serializable_cfg = make_json_serializable(cfg)
    with open(path, "w") as f:
        json.dump(serializable_cfg, f, indent=4)