import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from pathlib import Path




def show_image_with_bbox(row):
    img = cv2.imread(row['img_path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    x1, y1 = row['x_1'], row['y_1']
    x2, y2 = row['x_2'], row['y_2']

    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    ax.add_patch(rect)
    ax.set_title(
        f"Hazard: {row['hazard_flag']} | Type: {row['hazard_type_name']}"
    )
    plt.show()