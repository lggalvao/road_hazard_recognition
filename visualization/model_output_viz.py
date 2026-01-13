import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path



def plot_predictions_vs_gt(df, pred_col='hazard_pred'):
    plt.figure()
    plt.scatter(df['frame_n'], df[pred_col], label='Prediction', alpha=0.6)
    plt.scatter(df['frame_n'], df['hazard_flag'], label='Ground Truth', alpha=0.6)
    plt.legend()
    plt.title("Hazard Prediction vs Ground Truth")
    plt.show()