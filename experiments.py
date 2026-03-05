#["FocalLoss", "weighted_FocalLoss", "weighted_CELoss", "CELoss"]

EXPERIMENTS_0 = [
    ##BEST SINGLE IMAGE CNN_LSTM WITH LITERATURE CLASSES
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-4,
        "weight_decay": 0.0001,
        "sequence_stride": 6, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
    },
    
    #{
    #    "input_feature_type": "single_img_input",
    #    "use_object_visible_side": None,
    #    "use_rear_light_status": None,
    #    "model": "CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_1",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
    #    "stage": 0,
    #    "run_epoch_profile": False,
    #    "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
    #    "lr_cosine_t_max": 1,
    #    "lr_cosine_eta_min": 1e-6,
    #    "num_epochs": 1,
    #    "global_lr": True, 
    #    "optimizer": "AdamW",  #SGD, Adam, AdamW
    #    "loss_function": "FocalLoss",
    #    "amp_enabled": True,
    #    "batch_size": 32,
    #    "freeze_strategy": None,  #head, partial, full
    #    "dropout_cnn_dynamic": 0.0,
    #    "dropout_cnn": 0.5,
    #    "dropout_pre_attention": 0.0,
    #    "dropout_fc": 0.5,
    #    "backbone_lr": 2e-6,
    #    "head_lr": 2e-6,
    #    "weight_decay": 0.01,
    #    "sequence_stride": 6, 
    #    "cached_dataset": False,
    #    "comments": "sequence_stride and backbone_lr"
    #},

]


EXPERIMENTS_1 = [

    #{
    #    "input_feature_type": "single_img_input",
    #    "use_object_visible_side": None,
    #    "use_rear_light_status": None,
    #    "model": "CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_1",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
    #    "stage": 0,
    #    "run_epoch_profile": False,
    #    "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
    #    "lr_cosine_t_max": 1,
    #    "lr_cosine_eta_min": 1e-6,
    #    "num_epochs": 30,
    #    "global_lr": True, 
    #    "optimizer": "AdamW",  #SGD, Adam, AdamW
    #    "loss_function": "FocalLoss",
    #    "amp_enabled": True,
    #    "batch_size": 32,
    #    "freeze_strategy": None,  #head, partial, full
    #    "dropout_cnn_dynamic": 0.0,
    #    "dropout_cnn": 0.5,
    #    "dropout_pre_attention": 0.0,
    #    "dropout_fc": 0.5,
    #    "backbone_lr": 1e-6,
    #    "head_lr": 1e-6,
    #    "weight_decay": 0.01,
    #    "sequence_stride": 1, 
    #    "cached_dataset": False,
    #    "comments": "sequence_stride and backbone_lr"
    #},
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-6,
        "head_lr": 2e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-6,
        "head_lr": 3e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-6,
        "head_lr": 4e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 5e-6,
        "head_lr": 5e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 6e-6,
        "head_lr": 6e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 7e-6,
        "head_lr": 7e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 8e-6,
        "head_lr": 8e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 9e-6,
        "head_lr": 9e-6,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
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
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "AdamW",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 1e-5,
        "head_lr": 1e-5,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
    },
]