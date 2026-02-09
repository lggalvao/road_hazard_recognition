import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
import logging

logger = logging.getLogger("hazard_recognition")

def load_raw_data(cfg) -> pd.DataFrame:
    """Load the main CSV containing all video samples."""

    df = pd.read_csv(cfg.data.dataset_csv_file_path)
    df["img_path"] = df["img_path"].apply(
        lambda x: x.replace("/EEdata/bllg002/", cfg.system.root).replace("C:/", cfg.system.root)
    )
    return df


def add_no_hazard_samples(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Append manually checked 'no hazard' samples if enabled."""

    for csv in [cfg.data.no_hazard_samples_train_csv_file_path, cfg.data.no_hazard_samples_test_csv_file_path]:
        csv = Path(csv)
        if csv.exists():
            extra = pd.read_csv(csv)
            df = pd.concat([df, extra], ignore_index=True)
    return df


def save_or_load_normalization(cfg, df: pd.DataFrame, phase: str, keys: List[str]) -> Dict[str, float]:
    """Save normalization stats during training, or load for val/test."""
    file_path = "./normalization_info.json"
    if phase == "train" and not(cfg.data.load_normalization):
        info = compute_normalization_info(df, keys)
        with open(file_path, "w") as f:
            json.dump(info, f, indent=4)
    else:
        logger.info("Loading Normalization")
        with open(file_path, "r") as f:
            info = json.load(f)
    return info


def compute_normalization_info(df: pd.DataFrame, keys: List[str]) -> Dict[str, float]:
    """Compute mean/std normalization values for the given keys."""
    logger.info("Computing Normalization")
    return {f"{k}_{sfx}": getattr(df[k], sfx)() for k in keys for sfx in ("mean", "std")}


def apply_normalization(df: pd.DataFrame, info: Dict[str, float], keys: List[str]) -> pd.DataFrame:
    """Normalize numerical columns using pre-computed stats."""
    for k in keys:
        mean, std = info[f"{k}_mean"], info[f"{k}_std"]
        df[f"{k}_norm"] = (df[k] - mean) / (std + 1e-8)
    return df
