import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path



def plot_hazard_over_time(df: pd.DataFrame, video_id):
    subset = df[df['video_n'] == video_id]

    plt.figure()
    plt.plot(subset['frame_n'], subset['hazard_flag'])
    plt.title(f"Hazard Flag Over Time (Video {video_id})")
    plt.xlabel("Frame Number")
    plt.ylabel("Hazard Flag")
    plt.show()


def plot_speed(df):
    plt.scatter(df['frame_n'], df['speed'], alpha=0.3)
    plt.title("XC Speed Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Speed")
    plt.show()



