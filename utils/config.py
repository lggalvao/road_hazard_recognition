from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch
import json


@dataclass
class SystemConfig:
    multi_gpu: bool = False
    device: Optional[str] = None
    seed: int = 100
    root: str = "C:/"  # C:/ or /data/home/r2049970/ or /home/ubuntu/


@dataclass
class DataConfig:
    input_feature_type: str = "explicit_feature"
    input_img_type1: Optional[str] = "img_local_context_ROI_1"
    input_img_type2: Optional[str] = None
    input_img_type3: Optional[str] = None
    num_of_input_imgs: Optional[int] = None
    input_img_resize: List[int] = field(default_factory=lambda: [224, 224])
    sequence_stride: int = 1
    dataset_trim: int = 500
    with_no_hazard_samples_flag: bool = False
    num_of_no_hazard_samples_train: int = 250
    num_of_no_hazard_samples_test: int = 50
    split_seed: int = 250
    
    train_videos_number: List[float] = field(default_factory=lambda: [4.0, 296.0])
    test_videos_number: List[float] = field(default_factory=lambda: [335.0, 30.0])
    test_samples_per_calss: Dict[str, float] = field(default_factory=dict)
    
    train_total_video_samples: int = 0
    train_video_samples_per_class: Dict[str, float] = field(default_factory=dict)
    train_total_sequences: int = 0
    train_sequences_per_class: Dict[str, float] = field(default_factory=dict)
    
    test_total_video_samples: int = 0
    test_video_samples_per_class: Dict[str, float] = field(default_factory=dict)
    test_total_sequences: int = 0
    test_sequences_per_class: Dict[str, float] = field(default_factory=dict)
    
    img_shape: List[int] = field(default_factory=lambda: [224, 224])
    #num_dynamic_features: int = 13
    load_normalization: bool = False
    split_dataset: str = True
    dataset_folder_path: str = ""
    train_csv_set_output_path: str = ""
    test_csv_set_output_path: str = ""
    dataset_event_time_csv_file_path: str = ""
    dataset_csv_file_path: str = ""
    saved_dataloader: bool = True
    num_workers: int = 0



@dataclass
class ModelConfig:
    model: str = "Embedding_Temporal_LSTM"
    cnn_model: Optional[str] = None
    cnn_pretrained: bool = False
    freeze_strategy: str = "head"
    encoder_type: str = "LSTM"
    enc_hidden_size: int = 112
    enc_input_seq_length: int = 13
    enc_layers_num: int = 1
    bi_directional: bool = False
    output_embedding_size: int = 112
    dropout_embedding_feature: float = 0.25
    fc_output_size: int = 128
    dropout1: float = 0.8
    num_classes: int = 10
    classes_type: str = "motion_towards"
    classes_name: List[str] = field(
        default_factory=lambda: [
            "object_crossing", "object_emerging", "object_meeting", "right_cut_in",
            "left_cut_in", "object_turning", "object_stopping", "object_reversing",
            "object_coming_out", "object_pulling_up"
        ]
    )
    object_visible_side: bool = True
    tailight_status: bool = True
    num_heads: int = 2
    num_object_types: int = 0
    num_visible_sides: int = 0
    num_tailight_statuses: int = 0
    num_kinematic_features: int = 0
    num_bbox_features: int = 0
    


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 30
    learning_rate: float = 1e-4
    cnn_lr: float = 1e-4
    optimizer: str = "Adam"
    weight_decay: float = 0.0006
    clip_grad: float = 5.0
    patience: int = 10
    ts_augme: bool = False
    step_size: int = 16
    gamma: float = 0.1
    momentum: float = 0.9
    stage: int = 1


@dataclass
class LossConfig:
    loss_function: str = "weighted_CELoss"
    focal_loss_gamma: float = 1.0
    class_weights: Optional[List[float]] = field(
        default_factory=lambda: [0.9180, 1.4963, 1.8604, 0.6465, 0.6390,
                             0.6114, 0.7189, 1.1274, 3.1453, 2.7636]
    )


@dataclass
class LoggingConfig:
    test_name: str = "explicit_feature_0"
    file_name: str = "explicit_feature"
    log_dir: str = "results/"
    comments: Optional[str] = None
    results_csv_file_path: str = ""
    wandb_entity: str = "ls_galvao-brunel-university-london"
    wandb_project: str = "HAZARD_LSTM"
    wandb_run_name: str = ""
    pred_save_wrong: bool = False
    pred_save_frame_flag: bool = False

@dataclass
class DebuggingConfig:
    transformed_input_img_dir: str = ""

@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debugging: LoggingConfig = field(default_factory=DebuggingConfig)



def load_config(path: str) -> Config:
    
    with open(path, "r") as f:
        cfg_data = json.load(f)
    
    cfg_data["loss"]["class_weights"] = cfg_data["loss"]["class_weights"]

    return Config(
        system=SystemConfig(**cfg_data["system"]),
        data=DataConfig(**cfg_data["data"]),
        model=ModelConfig(**cfg_data["model"]),
        training=TrainingConfig(**cfg_data["training"]),
        loss=LossConfig(**cfg_data["loss"]),
        logging=LoggingConfig(**cfg_data["logging"]),
    )


def save_config(cfg: Config, path: str):
    """Save nested dataclass config to JSON safely."""
    serializable_cfg = make_json_serializable(cfg)
    with open(path, "w") as f:
        json.dump(serializable_cfg, f, indent=4)


def build_paths(cfg: Config):
    root = cfg.system.root.rstrip("/") + "/"

    cfg.data.dataset_folder_path = root + "Projects/RoadHazardDataset_OK/frame_sequences/"
    cfg.data.dataset_csv_file_path = cfg.data.dataset_folder_path  + "/all_roadHazardDataset_videos.csv"
    
    cfg.data.no_hazard_samples_train_csv_file_path = cfg.data.dataset_folder_path + "manually_checked_no_hazard_samples_train.csv"
    cfg.data.no_hazard_samples_test_csv_file_path = cfg.data.dataset_folder_path + "manually_checked_no_hazard_samples_test.csv"
    
    cfg.data.train_csv_set_output_path = cfg.data.dataset_folder_path + 'roadHazardDataset_train_set.csv'
    cfg.data.test_csv_set_output_path = cfg.data.dataset_folder_path + 'roadHazardDataset_test_set.csv'
    cfg.data.dataset_event_time_csv_file_path = root + "Projects/hazard_samples_preprocessing/hazard_samples_info.csv"

    return cfg