EXPERIMENTS_0 = [
    {
        "input_feature_type": "single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128, 
        "lstm_dropout": 0.0, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 0.0,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1.6,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.6,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128, 
        "lstm_dropout": 0.0, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 0.0,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1.8,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.6,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128, 
        "lstm_dropout": 0.0, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 0.0,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2.0,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.6,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128, 
        "lstm_dropout": 0.0, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 0.0,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2.2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.6,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 2,
        "enc_hidden_size": 192, 
        "lstm_dropout": 0.5, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 0.0,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.6,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.0015,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "GPT"
    },


"""    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  #Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 15,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 3,
        "amp_enabled": True,
        "batch_size": 24,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.0,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": None,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": None
    },
        {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  #Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 15,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 3,
        "amp_enabled": True,
        "batch_size": 24,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.1,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": None,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": None
    },
        {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  #Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 15,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 24,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.2,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": None,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": None
    },
        {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  #Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 15,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 24,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.4,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": None,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": None
    },
        {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  #Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 15,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 24,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.6,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": None,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": None
    },
"""
]


EXPERIMENTS_1 = [
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",
    
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
    
        "enc_input_seq_length": 20,
        "enc_layers_num": 2,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.3,
    
        "classes_type": "literature_classes",
    
        "stage": 0,
        "run_epoch_profile": False,
    
        "lr_scheduler": "CosineAnnealingLRWarmUp",
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 30,
        "lr_cosine_eta_min": 5e-7,
    
        "num_epochs": 30,
        "global_lr": False,
    
        "optimizer": "AdamW",
    
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
    
        "amp_enabled": True,
        "batch_size": 24,
    
        "freeze_strategy": None,
    
        "dropout_cnn_dynamic": 0.1,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
    
        "backbone_lr": 8e-5,
        "head_lr": 3e-4,
        "weight_decay": 5e-4,
    
        "sequence_stride": 4,
        "cached_dataset": False,
    
        "comments": "deeper_lstm + correct_cosine + adamw + longer_seq"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "batch_size"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-5,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "lr_cosine_eta_min"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1.5,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1.6,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1.8,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "enc_hidden_size": 128,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2.0,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "focal_loss_gamma"
    },
]







"""
##BEST SINGLE IMAGE CNN_LSTM WITH LITERATURE CLASSES: 85.67
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0,
        "classes_type": "literature_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-6,
        "num_epochs": 30,
        "global_lr": False, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": False,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-4,
        "head_lr": 2e-5,
        "weight_decay": 0.00018,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "sequence_stride and backbone_lr"
    },
"""

"""
##BEST SINGLE IMAGE CNN_LSTM WITH ALL CLASSES: 64.76
    {
        "input_feature_type": "single_img_input",
        "use_object_visible_side": None,
        "use_rear_light_status": None,
        "model": "CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 1,
        "lstm_dropout": 0.0, 
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 10,
        "global_lr": True, 
        "optimizer": "Adam",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": True,
        "batch_size": 32,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": None,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": None,
        "dropout_fc": 0.5,
        "backbone_lr": 2e-6,
        "head_lr": None,
        "weight_decay": 0.01,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": None
    },
"""


"""
##BEST Explicit Feature Embedding_Temporal_LSTM WITH LITERATURE CLASSES: 82.6
    {
        "input_feature_type": "explicit_feature",  #explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input"
        "use_object_visible_side": False,
        "use_rear_light_status": False,
        "model": "Embedding_Temporal_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
        "input_img_type1": None,
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "enc_layers_num": 2,
        "lstm_dropout":0.0,
        "classes_type": "all_classes",  #motion_towards, all_classes, literature_classes
        "stage": 0,
        "run_epoch_profile": False,
        "lr_scheduler": "CosineAnnealingLRWarmUp",  #StepLR, CosineAnnealingLR, CosineAnnealingLRWarmUp
        "step_size": None,
        "gamma": None,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 1e-5,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 1,
        "amp_enabled": False,
        "batch_size": 64,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 3e-3,
        "head_lr": 3e-3,
        "weight_decay": 0.0009,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": None
    },
"""