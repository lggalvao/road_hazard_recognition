from .preprocessing import *
from typing import List, Dict
import pandas as pd
import numpy as np
from utils.visualization import debug_observation_sequence
from utils.timing import timeit



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
    """
    Build fixed-length temporal sequences from a prepared dataframe.
    Works with 'all_together_norm' and multiple camera inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - 'video_n'
        - 'frame_id'
        - 'all_together_norm' (list or np.ndarray)
        - 'all_together' (optional, raw)
        - 'img_path_root_hist' or 'input_img_path_hist' (if image model)
    seq_len : int
        Number of frames per sequence.
    stride : int
        Step between starting frames.
    use_multicam : bool
        If True, expects additional input_img_path_hist_2, _3.
    num_input_imgs : int
        How many input cameras (1, 2, or 3).
    label_key : str
        Column for frame-level label.

    Returns
    -------
    list[dict]
        Each dict corresponds to one temporal sequence.
    """
    sequences = []

    for vid, group in df.groupby("video_n"):

        img_paths_2 = []
        img_paths_3 = []
        group = group.sort_values("frame_n").reset_index(drop=True)
        
        group = group[group['frame_n'] >= group.start_frame[0]]  #Only consider frames that have the hazard
        group = group.reset_index(drop = True)
        
        group = group[group['frame_n'] <= group.end_frame[0]]  # Only consider frames that have the hazard
        group = group.reset_index(drop = True)
        
        # Convert columns to lists/arrays
        object_detected = group["object_detected"].to_list()
        object_type_feats = group["object_type_consecutive"].to_list()
        object_visible_side_int_feats = group["object_visible_side_int"].to_list()
        tailight_status_int_feats = group["tailight_status_int"].to_list()
        categorical_feats = group["categorical"].to_list()
        kinematic_feats = group["kinematic"].to_list()
        bbox_feats = group["bbox"].to_list()
        norm_feats = group["all_together_norm"].to_list()
        raw_feats = group["all_together"].to_list()
        true_hazard_enc = group["true_hazard_enc"].to_list()
        hazard_type_name = group["hazard_type_name"].to_list()

        # Handle image paths (single or multi-camera)
        if group['hazard_type_name'].unique() == 'no_hazard' and cfg.data.with_no_hazard_samples_flag == True:

            if cfg.system.root == 'C:/':
                group['input_img'] = group.apply(lambda row: row.img_path.replace(row.img_path[46:], 'no_hazard_samples/' + str(int(row.video_n)).zfill(4) + '/img_original_size/' + str(int(row.frame_n)).zfill(5) + '.png' ), axis=1)
                img_path_root_hist = group.img_path.apply(lambda x: x[:68]).to_list()
            
            elif cfg.system.root == '/EEdata/bllg002/':
                group['input_img'] = group.apply(lambda row: row.img_path.replace(row.img_path[59:], 'no_hazard_samples/' + str(int(row.video_n)).zfill(4) + '/img_original_size/' + str(int(row.frame_n)).zfill(5) + '.png' ), axis=1)
                img_path_root_hist = group.img_path.apply(lambda x: x[:85]).to_list()
            
            group['input_img'] = group.input_img.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type1))

        else:
            group['input_img'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type1))
            img_path_root_hist = group.img_path.to_list()

        if cfg.data.input_feature_type == 'multi_img_input' or cfg.data.input_feature_type == 'explicit_and_multi_img_input':
            group['input_img_2'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type2))
            group['input_img_2'] = group.input_img_2.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
            img_paths_2 = group["input_img_2"].to_list()
            
            if cfg.data.num_of_input_imgs == 3:
                group['input_img_3'] = group.img_path.apply(lambda x: x.replace('img_original_size', cfg.data.input_img_type3))
                group['input_img_3'] = group.input_img_3.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
                img_paths_3 = group["input_img_3"].to_list()

        group['input_img'] = group.input_img.apply(lambda x: x.replace('\\', '/'))
        group['input_img'] = group.input_img.apply(lambda x: x.replace('C:/Projects/RoadHazardDataset/frame_sequences/', cfg.data.dataset_folder_path))
        # Iterate temporal windows. Note: need to decrese -1 from the seq length to consider the last frame as prediction
        for start in range(0, len(group) - (seq_len +1), stride):

            end = start + seq_len

            seq_dict = {
                "video_n": int(vid),
                "frame_n": list(group["frame_n"].iloc[start:end]),
                "all_together_norm_hist": np.array(norm_feats[start:end]),
                "object_type_feats_hist": np.array(object_type_feats[start:end]),
                "object_visible_side_int_feats_hist": np.array(object_visible_side_int_feats[start:end]),
                "tailight_status_int_feats_hist": np.array(tailight_status_int_feats[start:end]),
                "categorical_hist": np.array(categorical_feats[start:end]),
                "kinematic_hist": np.array(kinematic_feats[start:end]),
                "bbox_hist": np.array(bbox_feats[start:end]),
                "object_detected_hist": np.array(object_detected[start:end]),
                "all_together_hist": np.array(raw_feats[start:end]),
                "true_hazard": np.array(hazard_type_name[end+1]),
                "true_hazard_enc": np.array(true_hazard_enc[end+1]),
                "start_frame_hist": [int(group["frame_n"].iloc[start])],
                "end_frame_hist": [int(group["frame_n"].iloc[end - 1])],
                "hazard_name_hist": list(group.get("hazard_type_name", ["unknown"] * seq_len)[
                    start:end
                ]),
                "img_path_root_hist": img_path_root_hist[start:end],
                "original_frame_path_hist": group.get("img_path", [None] * seq_len)[
                    start:end
                ].to_list(),
                "input_img_path_hist": group.get("input_img", [None] * seq_len)[
                    start:end
                ].to_list(),
            }
            

            # Add optional multi-camera paths
            if img_paths_2:
                seq_dict["input_img_path_hist_2"] = img_paths_2[start:end]
            if img_paths_3:
                seq_dict["input_img_path_hist_3"] = img_paths_3[start:end]

            #if phase == "val":
            #    debug_observation_sequence(seq_dict['all_together_hist'], seq_dict['input_img_path_hist'], seq_dict['video_n'], seq_dict['frame_n'], seq_dict['start_frame_hist'], seq_dict['end_frame_hist'], seq_dict['hazard_name_hist'])
            
            sequences.append(seq_dict)

    return sequences
