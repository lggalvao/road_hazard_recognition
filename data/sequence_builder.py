from .preprocessing import *
from typing import List, Dict
import pandas as pd
import numpy as np
from utils.visualization import debug_observation_sequence
from utils.timing import timeit
import torch

pd.set_option('display.max_colwidth', None)

"""
Temporal Attention LSTM was chosen over Trasnformer because:
    Given the short temporal horizon and limited dataset size,
    an LSTM with temporal attention provided superior inductive
    bias and more stable optimization.
"""
@timeit
def create_temporal_sequences(
    cfg,
    df,
    seq_len: int,
    stride: int,
    label_key: str = "hazard_type_int",
    phase: str = "train"
):

    sequences = []

    # Pre-sort once globally (instead of per group)
    df = df.sort_values(["video_n", "frame_n"])

    for vid, group in df.groupby("video_n", sort=False):

        # Filter once
        start_f = group.start_frame.iloc[0]
        end_f = group.end_frame.iloc[0]
        
        group = group[(group["frame_n"] >= start_f) &
                      (group["frame_n"] <= end_f)]
        
        if len(group) < seq_len + 2:
            continue

        # ===============================
        # ---- Vectorized string ops ----
        # ===============================
        img_path = group["img_path"].astype(str)
        
        is_no_hazard = (
            group["hazard_type_name"].nunique() == 1 and
            group["hazard_type_name"].iloc[0] == "no_hazard" and
            cfg.data.with_no_hazard_samples_flag
        )
        
        if cfg.data.input_feature_type != "explicit_feature":
            if is_no_hazard:
            
                video_str = str(int(vid)).zfill(4)
                frame_str = group["frame_n"].astype(int).astype(str).str.zfill(5)
            
                #print(cfg.system.root)
                #print("img_path", img_path)
                #
                #if cfg.system.root == 'C:/':
                #    base_root = img_path.str.slice(0, 46)
                #
                #elif cfg.system.root == '/data/home/r2049970/':
                #    
                #    base_root = img_path.str.slice(0, 63)
                #    print("base_root", base_root)
                #
                #else:
                #    raise ValueError("Unsupported root")
            
                input_img = (
                    cfg.data.dataset_folder_path +
                    "no_hazard_samples/" +
                    video_str +
                    f"/{cfg.data.input_img_type1}/" +
                    frame_str +
                    ".png"
                )
            
                img_path_root_hist = img_path.to_numpy()
            
            else:
                input_img = img_path.str.replace(
                    "img_original_size",
                    cfg.data.input_img_type1,
                    regex=False
                )
            
                img_path_root_hist = img_path.to_numpy()
        else:
            input_img = img_path

        # Normalize slashes + dataset root
        print("input_img", input_img)
        input_img = input_img.str.replace("\\", "/", regex=False)
        input_img = input_img.str.replace(
            "C:/Projects/RoadHazardDataset/frame_sequences/",
            cfg.data.dataset_folder_path,
            regex=False
        )
        
        input_img = input_img.str.replace(
            "/data/home/r2049970/Projects/RoadHazardDataset/frame_sequences/",
            cfg.data.dataset_folder_path,
            regex=False
        )
        
        # Multi-camera handling
        img_paths_2 = None
        img_paths_3 = None

        if cfg.data.input_feature_type in (
            "multi_img_input",
            "explicit_and_multi_img_input"
        ):
            img_paths_2 = img_path.str.replace(
                "img_original_size",
                cfg.data.input_img_type2,
                regex=False
            ).str.replace(
                "C:/Projects/RoadHazardDataset/frame_sequences/",
                cfg.data.dataset_folder_path,
                regex=False
            ).to_numpy()

            if cfg.data.num_of_input_imgs == 3:
                img_paths_3 = img_path.str.replace(
                    "img_original_size",
                    cfg.data.input_img_type3,
                    regex=False
                ).str.replace(
                    "C:/Projects/RoadHazardDataset/frame_sequences/",
                    cfg.data.dataset_folder_path,
                    regex=False
                ).to_numpy()

        # ===============================
        # ---- Convert columns ONCE ----
        # ===============================

        frame_n = group["frame_n"].to_numpy()

        object_detected = np.stack(group["object_detected"].values)
        object_type_feats = np.stack(group["object_type_consecutive"].values)
        object_visible_side = np.stack(group["object_visible_side_int"].values)
        tailight_status = np.stack(group["tailight_status_int"].values)
        categorical_feats = np.stack(group["categorical"].values)
        kinematic_feats = np.stack(group["kinematic"].values)
        bbox_feats = np.stack(group["bbox"].values)
        norm_feats = np.stack(group["all_together_norm"].values)
        raw_feats = np.stack(group["all_together"].values)
        true_hazard_enc = np.stack(group["true_hazard_enc"].values)
        true_hazard_enc = np.argmax(true_hazard_enc, axis=1).astype(np.int64)
        hazard_type_name = group["hazard_type_name"].to_numpy()

        input_img = input_img.to_numpy()
        original_img = img_path.to_numpy()

        n = len(group)

        # ===============================
        # ---- Sliding Window ----
        # ===============================
        for start in range(0, n - (seq_len + 1), stride):

            end = start + seq_len
            target_idx = end + 1

            seq_dict = {
                "video_n": int(vid),
                "frame_n": frame_n[start:end].tolist(),

                "all_together_norm_hist": norm_feats[start:end],

                "object_type_feats_hist": object_type_feats[start:end],

                "object_visible_side_int_feats_hist": object_visible_side[start:end],

                "tailight_status_int_feats_hist": tailight_status[start:end],

                "categorical_hist": categorical_feats[start:end],

                "kinematic_hist": kinematic_feats[start:end],

                "bbox_hist": bbox_feats[start:end],

                "object_detected_hist": object_detected[start:end],

                "all_together_hist": raw_feats[start:end],

                "true_hazard_enc": true_hazard_enc[target_idx],

                "true_hazard": hazard_type_name[target_idx],

                "start_frame_hist": [int(frame_n[start])],
                "end_frame_hist": [int(frame_n[end - 1])],

                "hazard_name_hist":
                    hazard_type_name[start:end].tolist(),

                "img_path_root_hist":
                    original_img[start:end].tolist(),

                "original_frame_path_hist":
                    original_img[start:end].tolist(),

                "input_img_path_hist":
                    input_img[start:end].tolist(),
            }

            if img_paths_2 is not None:
                seq_dict["input_img_path_hist_2"] = \
                    img_paths_2[start:end].tolist()

            if img_paths_3 is not None:
                seq_dict["input_img_path_hist_3"] = \
                    img_paths_3[start:end].tolist()
            
            #if phase == "val":
            #    debug_observation_sequence(seq_dict['all_together_hist'], seq_dict['input_img_path_hist'], seq_dict['video_n'], seq_dict['frame_n'], seq_dict['start_frame_hist'], seq_dict['end_frame_hist'], seq_dict['hazard_name_hist'])
            
            sequences.append(seq_dict)

    return sequences
