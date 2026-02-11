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
        
        if is_no_hazard:
        
            video_str = str(int(vid)).zfill(4)
            frame_str = group["frame_n"].astype(int).astype(str).str.zfill(5)
        
            if cfg.system.root == 'C:/':
                base_root = img_path.str.slice(0, 46)
        
            elif cfg.system.root == '/EEdata/bllg002/':
                base_root = img_path.str.slice(0, 59)
        
            else:
                raise ValueError("Unsupported root")
        
            input_img = (
                base_root +
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

        # Normalize slashes + dataset root
        input_img = input_img.str.replace("\\", "/", regex=False)
        input_img = input_img.str.replace(
            "C:/Projects/RoadHazardDataset/frame_sequences/",
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
        #true_hazard_enc = group["true_hazard_enc"].to_numpy()
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

                # Direct zero-copy → torch
                "all_together_norm_hist": torch.from_numpy(
                    norm_feats[start:end]
                ).float(),

                "object_type_feats_hist": torch.from_numpy(
                    object_type_feats[start:end]
                ).long(),

                "object_visible_side_int_feats_hist": torch.from_numpy(
                    object_visible_side[start:end]
                ).long(),

                "tailight_status_int_feats_hist": torch.from_numpy(
                    tailight_status[start:end]
                ).long(),

                "categorical_hist": torch.from_numpy(
                    categorical_feats[start:end]
                ).float(),

                "kinematic_hist": torch.from_numpy(
                    kinematic_feats[start:end]
                ).float(),

                "bbox_hist": torch.from_numpy(
                    bbox_feats[start:end]
                ).float(),

                "object_detected_hist": torch.from_numpy(
                    object_detected[start:end]
                ).float(),

                "all_together_hist": torch.from_numpy(
                    raw_feats[start:end]
                ).float(),

                "true_hazard_enc": torch.tensor(
                    true_hazard_enc[target_idx],
                    dtype=torch.long
                ),

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



#@timeit
#def create_temporal_sequences(
#    cfg,
#    df,
#    seq_len: int,
#    stride: int,
#    label_key: str = "hazard_type_int",
#    phase: str = "train"
#    ):
#    """
#    Build fixed-length temporal sequences from a prepared dataframe.
#    Works with 'all_together_norm' and multiple camera inputs.
#
#    Parameters
#    ----------
#    df : pd.DataFrame
#        Must contain columns:
#        - 'video_n'
#        - 'frame_id'
#        - 'all_together_norm' (list or np.ndarray)
#        - 'all_together' (optional, raw)
#        - 'img_path_root_hist' or 'input_img_path_hist' (if image model)
#    seq_len : int
#        Number of frames per sequence.
#    stride : int
#        Step between starting frames.
#    use_multicam : bool
#        If True, expects additional input_img_path_hist_2, _3.
#    num_input_imgs : int
#        How many input cameras (1, 2, or 3).
#    label_key : str
#        Column for frame-level label.
#
#    Returns
#    -------
#    list[dict]
#        Each dict corresponds to one temporal sequence.
#    """
#    sequences = []
#
#    for vid, group in df.groupby("video_n"):
#
#        img_paths_2 = []
#        img_paths_3 = []
#        group = group.sort_values("frame_n").reset_index(drop=True)
#        
#        group = group[group['frame_n'] >= group.start_frame[0]]  #Only consider frames that have the hazard
#        group = group.reset_index(drop = True)
#        
#        group = group[group['frame_n'] <= group.end_frame[0]]  # Only consider frames that have the hazard
#        group = group.reset_index(drop = True)
#        
#        # Convert columns to lists/arrays
#        object_detected = group["object_detected"].to_list()
#        object_type_feats = group["object_type_consecutive"].to_list()
#        object_visible_side_int_feats = group["object_visible_side_int"].to_list()
#        tailight_status_int_feats = group["tailight_status_int"].to_list()
#        categorical_feats = group["categorical"].to_list()
#        kinematic_feats = group["kinematic"].to_list()
#        bbox_feats = group["bbox"].to_list()
#        norm_feats = group["all_together_norm"].to_list()
#        raw_feats = group["all_together"].to_list()
#        true_hazard_enc = group["true_hazard_enc"].to_list()
#        hazard_type_name = group["hazard_type_name"].to_list()
#
#        # Handle image paths (single or multi-camera)
#        if group['hazard_type_name'].unique() == 'no_hazard' and cfg.data.with_no_hazard_samples_flag == True:
#
#            if cfg.system.root == 'C:/':
#                group['input_img'] = group.apply(lambda row: row.img_path.replace(row.img_path[46:], 'no_hazard_samples/' + str(int(row.video_n)).zfill(4) + '/img_original_size/' + str(int(row.frame_n)).zfill(5) + '.png' ), axis=1)
#                img_path_root_hist = group.img_path.apply(lambda x: x[:68]).to_list()
#            
#            elif cfg.system.root == '/EEdata/bllg002/':
#                group['input_img'] = group.apply(lambda row: row.img_path.replace(row.img_path[59:], 'no_hazard_samples/' + str(int(row.video_n)).zfill(4) + '/img_original_size/' + str(int(row.frame_n)).zfill(5) + '.png' ), axis=1)
#                img_path_root_hist = group.img_path.apply(lambda x: x[:85]).to_list()
#            
#            group['input_img'] = group.input_img.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type1))
#
#        else:
#            group['input_img'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type1))
#            img_path_root_hist = group.img_path.to_list()
#
#        if cfg.data.input_feature_type == 'multi_img_input' or cfg.data.input_feature_type == 'explicit_and_multi_img_input':
#            group['input_img_2'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type2))
#            group['input_img_2'] = group.input_img_2.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
#            img_paths_2 = group["input_img_2"].to_list()
#            
#            if cfg.data.num_of_input_imgs == 3:
#                group['input_img_3'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type3))
#                group['input_img_3'] = group.input_img_3.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
#                img_paths_3 = group["input_img_3"].to_list()
#
#        group['input_img'] = group.input_img.apply(lambda x: x.replace('\\', '/'))
#        group['input_img'] = group.input_img.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
#        # Iterate temporal windows. Note: need to decrese -1 from the seq length to consider the last frame as prediction
#        for start in range(0, len(group) - (seq_len +1), stride):
#
#            end = start + seq_len
#
#            seq_dict = {
#                "video_n": int(vid),
#                "frame_n": list(group["frame_n"].iloc[start:end]),
#                "all_together_norm_hist": np.array(norm_feats[start:end]),
#                "object_type_feats_hist": np.array(object_type_feats[start:end]),
#                "object_visible_side_int_feats_hist": np.array(object_visible_side_int_feats[start:end]),
#                "tailight_status_int_feats_hist": np.array(tailight_status_int_feats[start:end]),
#                "categorical_hist": np.array(categorical_feats[start:end]),
#                "kinematic_hist": np.array(kinematic_feats[start:end]),
#                "bbox_hist": np.array(bbox_feats[start:end]),
#                "object_detected_hist": np.array(object_detected[start:end]),
#                "all_together_hist": np.array(raw_feats[start:end]),
#                "true_hazard": np.array(hazard_type_name[end+1]),
#                "true_hazard_enc": np.array(true_hazard_enc[end+1]),
#                "start_frame_hist": [int(group["frame_n"].iloc[start])],
#                "end_frame_hist": [int(group["frame_n"].iloc[end - 1])],
#                "hazard_name_hist": list(group.get("hazard_type_name", ["unknown"] * seq_len)[
#                    start:end
#                ]),
#                "img_path_root_hist": img_path_root_hist[start:end],
#                "original_frame_path_hist": group.get("img_path", [None] * seq_len)[
#                    start:end
#                ].to_list(),
#                "input_img_path_hist": group.get("input_img", [None] * seq_len)[
#                    start:end
#                ].to_list(),
#            }
#            
#
#            # Add optional multi-camera paths
#            if img_paths_2:
#                seq_dict["input_img_path_hist_2"] = img_paths_2[start:end]
#            if img_paths_3:
#                seq_dict["input_img_path_hist_3"] = img_paths_3[start:end]
#
#            #if phase == "val":
#            #    debug_observation_sequence(seq_dict['all_together_hist'], seq_dict['input_img_path_hist'], seq_dict['video_n'], seq_dict['frame_n'], seq_dict['start_frame_hist'], seq_dict['end_frame_hist'], seq_dict['hazard_name_hist'])
#            
#            sequences.append(seq_dict)
#
#    return sequences

