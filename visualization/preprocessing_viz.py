import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path




def plot_bbox_size_distribution(df: pd.DataFrame):
    plt.figure()
    plt.hist(df['bbox_area'], bins=50)
    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Area")
    plt.ylabel("Frequency")
    plt.show()


def plot_object_positions(df: pd.DataFrame):
    plt.figure()
    plt.scatter(df['xc'], df['yc'], alpha=0.3)
    plt.title("Object Center Positions")
    plt.xlabel("xc")
    plt.ylabel("yc")
    plt.gca().invert_yaxis()
    plt.show()