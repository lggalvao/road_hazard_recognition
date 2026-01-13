# data/feature_builder.py
from typing import List
import numpy as np
import pandas as pd


raw_cols = [
    "object_type",
    "tailight_status_int",
    "object_visible_side_int",
    "xc", "yc", "w", "h",
    "x_1", "y_1", "x_2", "y_2"
]

kinematic = [
    "x_n", "y_n", "vx_n", "vy_n", "ax_n", "ay_n",
    "speed", "theta", "dtheta",
    "scale", "dscale", "aspect", "daspect",
    "border_dist"
]

bbox = ["w_n", "h_n", "bbox_area_n"]

categorical_base = ["object_type"]


def build_all_together_features(video_df: pd.DataFrame, args: dict) -> pd.DataFrame:
    df = video_df.copy()

    # ---- Ensure raw columns exist ----
    for c in raw_cols:
        if c not in df.columns:
            df[c] = 0.0

    raw_vals = df[raw_cols].astype(np.float32).to_numpy()
    df["all_together"] = raw_vals.tolist()

    # ---- Categorical features ----
    categorical = list(categorical_base)

    if args.get("object_visible_side", False):
        categorical.append("object_visible_side_int")

    if args.get("tailight_status", False):
        categorical.append("tailight_status_int")

    # ---- Normalized numeric features (ONLY numeric) ----
    norm_cols = kinematic + bbox + categorical

    for c in norm_cols:
        if c not in df.columns:
            df[c] = 0.0

    norm_vals = df[norm_cols].astype(np.float32).to_numpy()
    df["all_together_norm"] = norm_vals.tolist()

    # ---- Optional: keep explicit groups for debugging / augmentation ----
    df["categorical"] = df[categorical].astype(np.float32).to_numpy().tolist()
    df["kinematic"] = df[kinematic].astype(np.float32).to_numpy().tolist()
    df["bbox"] = df[bbox].astype(np.float32).to_numpy().tolist()

    return df
