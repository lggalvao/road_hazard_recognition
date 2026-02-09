from __future__ import print_function, division
from torch.utils.data import (
    Dataset,
    DataLoader
    )
import numpy as np
import torch
import pandas as pd
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import json
from torchvision import transforms
import torchvision.transforms.functional as TF
import tsaug
from tsaug.visualization import plot
import wandb
from data.numeric_temporal_transforms import (
    augment_numeric_timeseries
    )
from data.image_temporal_transforms import (
    augment_video_frames,
    temporal_jittering_fixed,
    temporal_time_warp,
    temporal_jitter
    )
from train.losses import prepare_sampler_and_weights_from_sequences
from pathlib import Path
from typing import Any, Dict
from .preprocessing import (
    load_raw_data,
    add_no_hazard_samples,
    save_or_load_normalization,
    apply_normalization,
)
from .sequence_builder import create_temporal_sequences
from .feature_builder import build_all_together_features
import logging
from utils.timing import timeit
import matplotlib.pyplot as plt
from utils.visualization import debug_input_img_sequence
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

logger = logging.getLogger("hazard_recognition")


@timeit
class RoadHazardDataset(Dataset):
    """
    PyTorch Dataset for the Road Hazard Recognition project.

    Args:
        cfg: Configuration object.
        dataset_split_path (str): Path to the CSV defining the dataset split.
        phase (str): One of ["train", "val", "test"].
    """

    # Keys to implement normalization
    NUMERIC_KEYS = [
        "x_n", "y_n",
        "w_n", "h_n",
        "bbox_area_n",
        "vx_n", "vy_n",
        "ax_n", "ay_n",
        "speed",
        "theta", "dtheta",
        "scale", "dscale",
        "aspect", "daspect",
        "border_dist"
    ]

    @timeit
    def __init__(self, cfg, dataset_split_path: str, phase: str):
        self.cfg = cfg
        self.phase = phase
        self.root = Path(cfg.system.root)
        self.t_h = cfg.model.enc_input_seq_length
        self.num_input_imgs = cfg.data.num_of_input_img
        self.input_feature_type = cfg.data.input_feature_type
        self.stride = cfg.data.sequence_stride
        self.input_img_resize = cfg.data.input_img_resize
        self.ts_augme = cfg.training.ts_augme
        self.model = cfg.model.model

        # ---- Load and prepare data ----
        data = load_raw_data(cfg)

        # This must be done before filtering the data becasue if using small number
        #of clsses might not get all the object classes
        cfg.model.num_object_types = len(data["object_type_consecutive"].unique())
        cfg.model.num_visible_sides = len(data["object_visible_side_int"].unique())
        cfg.model.num_tailight_statuses = len(data["tailight_status_int"].unique())
        cfg.model.emb_dim_object_type = 5
        cfg.model.emb_dim_visible_side = 2
        cfg.model.emb_dim_tailight_status = 5
        
        #if cfg.data.with_no_hazard_samples_flag:
        #    data = add_no_hazard_samples(cfg, data)
        #data.to_csv("./debug.csv", index=False)
        
         # ---- Keep only the target object ----
        data = data.loc[(data['ID'] == data['target_obj_id'])]
        data = data.reset_index(drop = True)

        # ---- Apply split ----
        split_df = pd.read_csv(dataset_split_path)
        data = self._filter_by_split(data, split_df)

        # ---- Normalize numeric columns ----
        norm_info = save_or_load_normalization(cfg, data, phase, self.NUMERIC_KEYS)
        data = apply_normalization(data, norm_info, self.NUMERIC_KEYS)
        
        # build features here (args could be cfg.model and cfg flags)
        args_dict = {
            "object_visible_side": cfg.model.object_visible_side,
            "tailight_status": cfg.model.tailight_status,
            "model": cfg.model.model,
        }
        
        data = build_all_together_features(
            data,
            args_dict,
        )
        
        label_to_index = {name: i for i, name in enumerate(cfg.model.classes_name)}
        indices = torch.tensor([label_to_index[x] for x in data.hazard_type_name])
        
        true_hazard_enc = torch.nn.functional.one_hot(
            indices,
            num_classes=cfg.model.num_classes
        )

        data.insert(len(data.columns)-1, "true_hazard_enc", true_hazard_enc.tolist())
        
        # ---- Create temporal sequences ----
        self.samples = create_temporal_sequences(
            cfg,
            df=data,
            seq_len=self.t_h,
            stride=self.stride,
            label_key="hazard_type_int",
            phase = self.phase
        )
        
        get_sequence_samples_info(
            self.cfg,
            self.phase,
            self.samples
        )


    def _filter_by_split(self, data: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows by the provided split CSV."""
        if "new_clip_name" in split_df.columns:
            valid_videos = set(split_df["new_clip_name"].unique())
            return data[data["video_n"].isin(valid_videos)].reset_index(drop=True)
        return data

    # -----------------------------
    # Standard Dataset API
    # -----------------------------
    def __len__(self) -> int:
        return len(self.samples)


    @timeit
    def __getitem__(self, index):
    
        if index >= len(self.samples):
            raise IndexError(
                f"Index {index} out of range for samples size {len(self.samples)}. Phase {self.phase}"
            )
        
        #Save some samples for debugging
        debug = False
        if self.phase == "train" and index in [0,1, 2, 3, 4, 5]:
            debug = True
    
        seq = self.samples[index]
    
        # ---------------------------------
        #   Image transforms (SAFE)
        # ---------------------------------
        if self.phase == "train":
            img_transforms = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
            ])
        else:
            img_transforms = transforms.ToTensor()
    
        # ---------------------------------
        #   Load image sequences
        # ---------------------------------
        img_tensors = []
    
        if self.input_feature_type != "explicit_feature":
    
            img_inputs = [seq["input_img_path_hist"]]
            if "input_img_path_hist_2" in seq:
                img_inputs.append(seq["input_img_path_hist_2"])
            if "input_img_path_hist_3" in seq:
                img_inputs.append(seq["input_img_path_hist_3"])
    
            for img_list in img_inputs:
                imgs = build_img_tensor(
                    self.cfg,
                    img_list,
                    img_transforms,
                    debug
                )
                img_tensors.append(imgs)  # [T, C, H, W]
    
            if self.model == "C3D":
                img_tensors = [t.permute(1, 0, 2, 3) for t in img_tensors]  # [C, T, H, W]
    
        # ---------------------------------
        #   Categorical features (INT)
        # ---------------------------------
        object_type = torch.as_tensor(
            np.array(seq["object_type_feats_hist"]), dtype=torch.long
        ).squeeze(0)
    
        object_visible_side = torch.as_tensor(
            np.array(seq["object_visible_side_int_feats_hist"]), dtype=torch.long
        ).squeeze(0)
    
        tailight_status = torch.as_tensor(
            np.array(seq["tailight_status_int_feats_hist"]), dtype=torch.long
        ).squeeze(0)
    
        # ---------------------------------
        #   Numeric features (FLOAT)
        # ---------------------------------
        kinematic = torch.as_tensor(
            np.array(seq["kinematic_hist"]), dtype=torch.float32
        ).squeeze(0)
    
        bbox = torch.as_tensor(
            np.array(seq["bbox_hist"]), dtype=torch.float32
        ).squeeze(0)
    
        # ---------------------------------
        #   Numeric augmentation (TRAIN ONLY)
        # ---------------------------------
        if self.phase == "train":
            kinematic, bbox = augment_numeric_timeseries(
                kinematic=kinematic,
                bbox=bbox,
                noise_std=0.01,
                scale_range=(0.95, 1.05),
                max_time_shift=0,  # MUST stay 0 (frame-aligned)
                debug=debug,
                sample_index=index,
            )
    
        # ---------------------------------
        #   Labels & metadata
        # ---------------------------------
        true_hazard_enc = torch.as_tensor(
            np.array(seq["true_hazard_enc"]), dtype=torch.long
        )
    
        frame_n = torch.as_tensor(
            seq["frame_n"], dtype=torch.long
        )
    
        missing_object_mask = torch.as_tensor(
            seq["object_detected_hist"], dtype=torch.float32
        )
    
        start_frame = torch.as_tensor(seq["start_frame_hist"], dtype=torch.long)
        end_frame = torch.as_tensor(seq["end_frame_hist"], dtype=torch.long)
    
        # ---------------------------------
        #   Sanity checks (IMPORTANT)
        # ---------------------------------
        T = self.cfg.model.enc_input_seq_length
    
        assert kinematic.shape[0] == T
        assert bbox.shape[0] == T
        assert object_type.shape[0] == T
        assert object_type.dtype == torch.long
        assert kinematic.dtype == torch.float32
    
        # ---------------------------------
        #   Return (NO FUSION HERE)
        # ---------------------------------
        return {
            "images": img_tensors,               # list of [T, C, H, W]
            "kinematic": kinematic,              # [T, K] 
            "bbox": bbox,                        # [T, B]
            "object_type": object_type,          # [T]
            "object_visible_side": object_visible_side,
            "tailight_status": tailight_status,
            "true_hazard_enc": true_hazard_enc,
            "frame_n": frame_n,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "missing_object_mask": missing_object_mask,
            "hazard_name": seq["hazard_name_hist"],
            "img_root": seq["img_path_root_hist"],
            "original_frame_paths": seq["original_frame_path_hist"],
        }



def prepare_inputs(batch, cfg):
    """
    Prepare model inputs dynamically based on:
    - The configured input feature type.
    - The number of image streams available in the batch.

    Parameters
    ----------
    batch : dict
        Structured batch from the dataloader.
    cfg : Config
        Global configuration object.

    Returns
    -------
    inputs : tuple
        Model input tensors (in correct order based on config).
    labels : Tensor
        Ground-truth labels.
    """
    device = cfg.system.device
    feature_type = cfg.data.input_feature_type

    # ========= Helper: Move tensors or lists of tensors to device =========
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, list):
            return [t.to(device) for t in x]
        return x

    # ========= Extract batch fields =========
    kinematic = move(batch.get("kinematic"))                # list(Tensor) or None
    bbox = move(batch.get("bbox"))                # list(Tensor) or None
    object_visible_side = move(batch.get("object_visible_side"))                # list(Tensor) or None
    tailight_status = move(batch.get("tailight_status"))                # list(Tensor) or None
    object_type = move(batch.get("object_type"))                # list(Tensor) or None
    
    missing_object_mask = move(batch.get("missing_object_mask")) #List of tensors 
    images = batch.get("images")                # list(Tensor) or None
    #features_norm = batch.get("features_norm")  # Tensor or None
    labels = move(batch["true_hazard_enc"])           # Tensor, always present

    # Normalize images structure
    if images is None:
        images = []
    else:
        images = move(images)
        if not isinstance(images, list):
            raise ValueError("Expected batch['images'] to be a list of image tensors.")

    num_imgs = len(images)

    # ========= INPUT TYPE DISPATCHING =========

    # ----- Explicit Features Only -----
    if feature_type == "explicit_feature":

        return (
            {
                "kinematic": kinematic,
                "bbox": bbox,
                "object_type": object_type,
                "object_visible_side": object_visible_side,
                "tailight_status": tailight_status,
                "missing_object_mask": missing_object_mask
            },
            labels
        )
        #(kinematic, bbox, object_visible_side, tailight_status, object_type, missing_object_mask), labels

    # ----- Single Image Input (automatically selects first image) -----
    elif feature_type == "single_img_input":
        if num_imgs < 1:
            raise ValueError("single_img_input requires at least 1 image.")
            
        return (
            {
                "images": images[0],
                "missing_object_mask": missing_object_mask
            },
            labels
            )
        #return (images[0], missing_object_mask), labels

    # ----- Multi-Image Input (automatically uses all images) -----
    elif feature_type == "multi_img_input":
        if num_imgs < 2:
            raise ValueError("multi_img_input requires multiple images (>=2).")
        return tuple(images, missing_object_mask), labels

    # ----- Explicit + Single Image -----
    elif feature_type == "explicit_and_single_img_input":
        if features_norm is None:
            raise ValueError("explicit_and_single_img_input requires features_norm.")
        if num_imgs < 1:
            raise ValueError("explicit_and_single_img_input requires at least 1 image.")
        return (move(features_norm), images[0], missing_object_mask), labels

    # ----- Explicit + Multi-Image (automatically uses all images) -----
    elif feature_type == "explicit_and_multi_img_input":
        if features_norm is None:
            raise ValueError("explicit_and_multi_img_input requires features_norm.")
        if num_imgs < 2:
            raise ValueError("explicit_and_multi_img_input requires multiple images (>=2).")
        return (move(features_norm), *images, missing_object_mask), labels

    # ----- Trajectory Model -----
    elif cfg.model.model == "Trajectory_Embedding_LSTM":
        if features_norm is None:
            raise ValueError("Trajectory_Embedding_LSTM requires features_norm.")
        return (move(features_norm), missing_object_mask), labels

    else:
        raise ValueError(f"Unsupported cfg.data.input_feature_type: {feature_type}")


def _set_class_weights(cfg):
    """Configure class weights deterministically."""
    weights_map = {
        "literature_classes": [1.3269, 0.9346, 0.9237, 0.8838, 1.0391],
        "motion_towards": [
            0.9180, 1.4963, 1.8604, 0.6465, 0.6390,
            0.6114, 0.7189, 1.1274, 3.1453, 2.7636
        ],
        "all_classes": [
            0.9761, 0.8735, 1.5910, 1.9782, 0.6875, 0.6795,
            0.6502, 0.7644, 1.1988, 0.6900, 1.3259, 3.3445,
            2.9386, 0.8018
        ],
    }

    if cfg.model.classes_type not in weights_map:
        raise ValueError(f"Unknown classes_type: {cfg.model.classes_type}")

    cfg.loss.class_weights = weights_map[cfg.model.classes_type]


def _build_dataloader(dataset, cfg, shuffle, drop_last, sampler):
    """Centralized DataLoader construction."""
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,#shuffle if cfg.data.sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers= False #cfg.data.num_workers > 0,
    )


@timeit
def create_or_load_dataset(cfg):
    """
    Best-practice dataset/dataloader creation.
    DataLoaders are ALWAYS rebuilt; only dataset state may be cached.
    """

    logger.info("----------- Loading Dataset -----------")

    # --------------------
    # TRAIN DATASET
    # --------------------
    trSet = RoadHazardDataset(
        cfg,
        cfg.data.train_csv_set_output_path,
        phase="train"
    )

    (cfg.loss.class_weights,
     cfg.data.sampler) = prepare_sampler_and_weights_from_sequences(
        cfg,
        trSet.samples,
        label_key='true_hazard',
        device=cfg.system.device,
        loss_function=cfg.loss.loss_function,
        classes_name = cfg.model.classes_name
    )
    
    logger.info(f"Class Name Order (Data Loading ): {cfg.model.classes_name}")
    logger.info(f"Encoded Example (Data Loading ): {trSet.samples[0]['true_hazard_enc'][0]}")
    logger.info(f"cfg.loss.class_weights {cfg.loss.class_weights}")
    logger.info(f"cfg.data.sampler {cfg.data.sampler}")
    
    trDataloader = _build_dataloader(
        trSet,
        cfg,
        shuffle=True,
        drop_last=False,
        sampler=cfg.data.sampler
    )

    logger.info(f"Train batches: {len(trDataloader)}")
    
    logger.info("\n----------- Train Statistics -----------")
    logger.info(f"Train videos total samples: {cfg.data.train_total_video_samples}")
    
    logger.info(f"Train videos samples per class:\n")
    for key, value in cfg.data.train_video_samples_per_class.items():
        logger.info(f"{key:<32} : {value:>8}   {(100* (value/cfg.data.train_total_video_samples)):.2f}%")
        
    logger.info(f"Train sequences total samples: {cfg.data.train_total_sequences}")
    logger.info(f"Train sequence per class:\n")
    for key, value in cfg.data.train_sequences_per_class.items():
        logger.info(f"{key:<32} : {value:>8}   {(100* (value/cfg.data.train_total_sequences)):.2f}%")


    # --------------------
    # VALIDATION / TEST DATASET
    # --------------------
    tsSet = RoadHazardDataset(
        cfg,
        cfg.data.test_csv_set_output_path,
        phase="val",
    )

    tsDataloader = _build_dataloader(
        tsSet,
        cfg,
        shuffle=False,
        drop_last=False,
        sampler=None
    )

    logger.info(f"Test batches: {len(tsDataloader)}")
    
    logger.info("\n----------- Test Statistics -----------")
    logger.info(f"Test videos total samples: {cfg.data.test_total_video_samples}")
    
    logger.info(f"Test videos samples per class:\n")
    for key, value in cfg.data.test_video_samples_per_class.items():
        logger.info(f"{key:<32} : {value:>8}   {100 * value / cfg.data.test_total_video_samples:6.2f}%")
        
    logger.info(f"Test sequences total samples: {cfg.data.test_total_sequences}")
    logger.info(f"Test sequence per class:\n")
    for key, value in cfg.data.test_sequences_per_class.items():
        logger.info(f"{key:<32} : {value:>8}   {(100* (value/cfg.data.test_total_sequences)):.2f}%")

    logger.info("Dataset initialization completed successfully")

    return {
        "train": trDataloader,
        "val": tsDataloader,   # note: val == test by design
        "test": tsDataloader,
    }


def create_or_load_dataset___(cfg):
    
    root_file_name = str(cfg.data.input_feature_type) + '_OH_' + str(cfg.model.enc_input_seq_length) + '_b_size' + str(cfg.training.batch_size)
    if cfg.data.saved_dataloader:
        print("Loading Dataset from Saved .pth File")
        trDataloader = torch.load('./output/dataloader/' + root_file_name + '_trDataloader.pth')
        print("len",len(trDataloader) * cfg.training.batch_size)
        print("trDataloader_len", len(trDataloader))

        tsDataloader = torch.load('./output/dataloader/' + root_file_name + '_tsDataloader.pth')
        print("len",len(tsDataloader) * cfg.training.batch_size)
        print("TestDataloader_len", len(tsDataloader))
        
        if cfg.model.classes_type == 'literature_classes':
            #Literature Classes
            cfg.loss.class_weights = [1.3269, 0.9346, 0.9237, 0.8838, 1.0391]
        
        elif cfg.model.classes_type == 'motion_towards':
            #Motion Towards
            cfg.loss.class_weights = [0.9180, 1.4963, 1.8604, 0.6465, 0.6390, 0.6114, 0.7189, 1.1274, 3.1453,  2.7636]
        
        elif cfg.model.classes_type == 'all_classes':
            #All Classes
            cfg.loss.class_weights = [0.9761, 0.8735, 1.5910, 1.9782, 0.6875, 0.6795, 0.6502, 0.7644, 1.1988, 0.6900, 1.3259, 3.3445, 2.9386, 0.8018]
        
        print("All dataset was loaded successfully")
    else:
    
        logger.info('\n-----------Loading Dataset-----------')
        trSet = RoadHazardDataset(cfg, cfg.data.train_csv_set_output_path, phase='train')
        trDataloader = DataLoader(trSet,batch_size=cfg.training.batch_size,num_workers=8, shuffle=False, drop_last = True, sampler=cfg.data.sampler) 
        torch.save(trDataloader, './output/dataloader/' + root_file_name + '_trDataloader.pth')
        print("\nNumber of Train Video Samples:", len(cfg.data.train_videos_number))
        print("Number of Train Sequence Samples:", len(trSet))
        print("Number of Train batches:", len(trDataloader))

        tsSet = RoadHazardDataset(cfg, cfg.data.test_csv_set_output_path, phase = 'val')
        tsDataloader = DataLoader(tsSet,batch_size=cfg.training.batch_size, shuffle=False, num_workers=8, drop_last = False)
        torch.save(tsDataloader, './output/dataloader/' + root_file_name + '_tsDataloader.pth')
        print("\nNumber of Test Video Samples:", len(cfg.data.test_videos_number))
        print("Number of Test Sequence Samples:", len(tsSet))
        print("Number of test batches:", len(tsDataloader))

    allsetDataloader = {'train':trDataloader,
                        'val'  :tsDataloader, #########################################note test and validation set are the same at the moment
                        'test' :tsDataloader}
    return allsetDataloader


def split_roadHazardDataset(cfg):
    print('\n-----------Splitting Dataset-----------')

    #right_cut_in: 113; object_stopping: 63; left_cut_in: 61; object_crossing: 60; object_turning: 51; object_hazard_light_on: 48; red_crossing_traffic_light: 40; object_meeting: 38; object_emerging: 37; pedestrian_near_parked_vehicles: 34; road_works: 34; object_reversing: 20; object_pulling_up: 14; object_coming_out: 11;

    dataset_timings_data_frame = pd.read_csv(cfg.data.dataset_event_time_csv_file_path)
    
    if cfg.model.classes_type == 'literature_classes':
        #5 classes: Classes that are studied by the literature
        hazard_list = ['right_cut_in', 'object_stopping', 'left_cut_in', 'object_crossing', 'object_turning']

    elif cfg.model.classes_type == '20_or_more':
        #12 classes: all classes that have >= 34 samples
        hazard_list = ['right_cut_in', 'object_stopping', 'left_cut_in', 'object_crossing', 'object_turning', 'object_hazard_light_on', 'red_crossing_traffic_light', 'object_meeting', 'object_emerging', 'pedestrian_near_parked_vehicles', 'road_works', 'object_reversing']

    elif cfg.model.classes_type == 'all_classes':
        #14 classes: [All available classes]
        hazard_list = ['right_cut_in', 'object_stopping', 'left_cut_in', 'object_crossing', 'object_turning', 'object_hazard_light_on', 'red_crossing_traffic_light', 'object_meeting', 'object_emerging', 'pedestrian_near_parked_vehicles', 'road_works', 'object_reversing', 'object_pulling_up', 'object_coming_out']

    elif cfg.model.classes_type == 'motion_towards':
        #10 classes: Classes that are trained by the explicit model only.
        hazard_list = ['right_cut_in', 'object_stopping', 'left_cut_in', 'object_crossing', 'object_turning', 'object_meeting', 'object_emerging', 'object_reversing', 'object_pulling_up', 'object_coming_out']
    
    road_hazard_train = pd.DataFrame()
    road_hazard_test = pd.DataFrame()
    #dataset_timings_data_frame = dataset_timings_data_frame[dataset_timings_data_frame['put_all_samples_together_done'] == True]
    for nu_hazard in dataset_timings_data_frame.hazard_type.unique():
        if nu_hazard in hazard_list: #and len(dataset_timings_data_frame[dataset_timings_data_frame['hazard_type'] == nu_hazard]) >= 18:
            temp_hazard_df = dataset_timings_data_frame[dataset_timings_data_frame['hazard_type'] == nu_hazard]

            if len(temp_hazard_df) > cfg.data.dataset_trim:
                temp_hazard_df = temp_hazard_df.sample(n=cfg.data.dataset_trim, random_state = 1)
            train, test = train_test_split(temp_hazard_df, test_size=0.20, random_state = 33)
            road_hazard_train = pd.concat([road_hazard_train, train])
            road_hazard_test = pd.concat([road_hazard_test, test])

    if cfg.data.with_no_hazard_samples_flag == True:
        # Merge the hazard samples with the no hazard samples
        road_hazard_train = road_hazard_train[['new_clip_name','hazard_type', 'hazard_type_int']]
        no_hazard_samples_train_set = pd.read_csv(cfg.data.dataset_folder_path + "manually_checked_no_hazard_samples_train.csv")
        no_hazard_samples_train_set = no_hazard_samples_train_set.video_n.unique()
        no_hazard_samples_train_set = pd.DataFrame(no_hazard_samples_train_set, columns=['new_clip_name']) 
        no_hazard_samples_train_set['hazard_type'] = 'no_hazard'
        no_hazard_samples_train_set['hazard_type_int'] = 18
        no_hazard_samples_train_set = no_hazard_samples_train_set.sample(n=cfg.data.num_of_no_hazard_samples_train, random_state = 1)
        road_hazard_train = pd.concat([road_hazard_train, no_hazard_samples_train_set])
        
        road_hazard_test = road_hazard_test[['new_clip_name','hazard_type', 'hazard_type_int']]
        no_hazard_samples_test_set = pd.read_csv(cfg.data.dataset_folder_path + "manually_checked_no_hazard_samples_test.csv")
        no_hazard_samples_test_set = no_hazard_samples_test_set.video_n.unique()
        no_hazard_samples_test_set = pd.DataFrame(no_hazard_samples_test_set, columns=['new_clip_name']) 
        no_hazard_samples_test_set['hazard_type'] = 'no_hazard'
        no_hazard_samples_test_set['hazard_type_int'] = 18
        no_hazard_samples_test_set = no_hazard_samples_test_set.sample(n=cfg.data.num_of_no_hazard_samples_test, random_state = 1)
        road_hazard_test = pd.concat([road_hazard_test, no_hazard_samples_test_set])
    
    cfg.model.num_classes = len(road_hazard_train.hazard_type.unique()) #+1 to consider the sequences that are not a hazard
    
    le = LabelEncoder()
    le.fit(road_hazard_train.hazard_type.unique())
    cfg.model.classes_name = le.classes_
    
    cfg.data.train_videos_number = road_hazard_train.new_clip_name.unique()
    cfg.data.train_total_video_samples = len(road_hazard_train.new_clip_name.unique())
    cfg.data.test_videos_number = road_hazard_test.new_clip_name.unique()
    cfg.data.test_total_video_samples = len(road_hazard_test.new_clip_name.unique())

    for hazard_name in road_hazard_train.hazard_type.unique():
        cfg.data.train_video_samples_per_class[hazard_name] = len(road_hazard_train[road_hazard_train['hazard_type'] == hazard_name])

    for hazard_name in road_hazard_test.hazard_type.unique():
        cfg.data.test_video_samples_per_class[hazard_name] = len(road_hazard_test[road_hazard_test['hazard_type'] == hazard_name])

    road_hazard_train.to_csv(cfg.data.train_csv_set_output_path, index=False)
    road_hazard_test.to_csv(cfg.data.test_csv_set_output_path, index=False)


def get_sequence_samples_info(cfg, phase, samples):
    
    labels = [sample["true_hazard"].item() for sample in samples]
    hazard_counts = Counter(labels)
    
    if phase == "train":
        cfg.data.train_sequences_per_class = dict(hazard_counts)
        cfg.data.train_total_sequences = sum(hazard_counts.values())
    
    if phase == "val":
        cfg.data.test_sequences_per_class = dict(hazard_counts)
        cfg.data.test_total_sequences = sum(hazard_counts.values())


@timeit
def safe_load_image(path, resize, transform):
    """Safely load an image, resize and apply transform."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize(resize, Image.BILINEAR)
    return transform(img)


@timeit
def build_img_tensor(cfg, seq_paths, transform, debug):
    # Pre-size Frames When Sequence Length is Fixed
    frames = [None] * len(seq_paths)
    for i, p in enumerate(seq_paths):
        with Image.open(p) as img:
            
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            if debug:
                debug_input_img_sequence(cfg, p, transform(img))

            frames[i] = transform(img)
    return torch.stack(frames)


@timeit
def augment_time_series(tensor_seq, augmenter=None, t_h=None, ts_augme=False):
    """Apply temporal/time-series augmentations if configured."""
    if ts_augme and augmenter is not None:
        try:
            return augment_video_frames(tensor_seq, augmenter)
        except Exception:
            # fallback if augmentation fails
            return tensor_seq
    return tensor_seq