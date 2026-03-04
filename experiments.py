#["FocalLoss", "weighted_FocalLoss", "weighted_CELoss", "CELoss"]

EXPERIMENTS_0 = [
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "lr_cosine_t_max": 0.2,
        "lr_cosine_eta_min": 1e-6,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 64,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-5,
        "head_lr": 3e-5,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": None
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 64,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-5,
        "head_lr": 3e-5,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "lr_cosine_t_max"
    },

]


EXPERIMENTS_1 = [

    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "lr_cosine_t_max": 1.0,
        "lr_cosine_eta_min": 1e-6,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 64,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-5,
        "head_lr": 3e-5,
        "weight_decay": 0.02,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "weight decay"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "lr_cosine_t_max": 1.0,
        "lr_cosine_eta_min": 1e-6,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 64,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-5,
        "head_lr": 3e-5,
        "weight_decay": 0.05,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "weight decay"
    },
    
    #{
    #    "input_feature_type": "explicit_and_single_img_input",  #explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input, 
    #    "use_object_visible_side": None,
    #    "use_rear_light_status": None,
    #    "model": "Embedding_CNN_LSTM",  # 
    #    "input_img_type1": "img_local_context_ROI_1",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
    #    "global_lr": True, 
    #    "optimizer": "SGD",  #SGD, Adam, AdamW
    #    "loss_function": "FocalLoss",
    #    "amp_enabled": False,
    #    "batch_size": 32,
    #    "freeze_strategy": None,  #head, partial, full
    #    "dropout_cnn_dynamic": 0.0,
    #    "dropout_cnn": 0.5,
    #    "dropout_pre_attention": 0.0,
    #    "dropout_fc": 0.5,
    #    "backbone_lr": 4e-4,
    #    "head_lr": 4e-4,
    #    "weight_decay": 0.0001,
    #    "sequence_stride": 4, 
    #    "cached_dataset": False,
    #    "comments": None
    #},
    #    {
    #    "input_feature_type": "explicit_feature",
    #    "use_object_visible_side": True,
    #    "use_rear_light_status": True,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "lr_scheduler": "StepLR",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
    #    "global_lr": True, 
    #    "optimizer": "SGD",  #SGD, Adam, AdamW
    #    "loss_function": "FocalLoss",
    #    "amp_enabled": False,
    #    "batch_size": 64,
    #    "freeze_strategy": None,  #head, partial, full
    #    "dropout_cnn_dynamic": 0.0,
    #    "dropout_cnn": 0.0,
    #    "dropout_pre_attention": 0.0,
    #    "dropout_fc": 0.5,
    #    "backbone_lr": 1e-04,
    #    "head_lr": 1e-04,
    #    "weight_decay": 0.0009,
    #    "sequence_stride": 1, 
    #    "cached_dataset": False,
    #    "comments": None
    #},
    #{
    #    "input_feature_type": "explicit_feature",
    #    "use_object_visible_side": True,
    #    "use_rear_light_status": True,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "lr_scheduler": "StepLR",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
    #    "global_lr": True, 
    #    "optimizer": "Adam",  #SGD, Adam, AdamW
    #    "loss_function": "FocalLoss",
    #    "amp_enabled": False,
    #    "batch_size": 64,
    #    "freeze_strategy": None,  #head, partial, full
    #    "dropout_cnn_dynamic": 0.0,
    #    "dropout_cnn": 0.0,
    #    "dropout_pre_attention": 0.0,
    #    "dropout_fc": 0.5,
    #    "backbone_lr": 1e-04,
    #    "head_lr": 1e-04,
    #    "weight_decay": 0.0009,
    #    "sequence_stride": 2, 
    #    "cached_dataset": False,
    #    "comments": None
    #},

]