import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path




def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def keep_target_objects(df):
    """
    Keep only rows corresponding to the target object.
    """
    target_df = df[
        (df["ID"] == df["target_obj_id"]) &
        (df["object_detected"] == 1)
    ].copy()

    return target_df