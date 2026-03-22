EXPERIMENTS_0 = [

    #Single Image Literature Classes
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
        "focal_loss_gamma": 2,
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
        "comments": "best"
    },
    
    # Single Image All Classes
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
        "focal_loss_gamma": 2,
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
        "comments": "best"
    },
]


EXPERIMENTS_1 = [
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": False,
        "use_rear_light_status": False,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 40,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 40,
        "global_lr": True, 
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
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "best"
    },
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": False,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 40,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 40,
        "global_lr": True, 
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
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0001,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "best"
    },
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 40,
        "gamma": 0.01,
        "lr_cosine_t_max": 1,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 40,
        "global_lr": True, 
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
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "best"
    },
    
    # Explicit + single image literature classes
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": False,
        "use_rear_light_status": False,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 22,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.6,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.6,
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0015,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "best"
    },
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": False,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "best"
    },
    # tryinf with cnn lr nd fc lr 0.6
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": False,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.6,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.6,
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.0015,
        "sequence_stride": 1, 
        "cached_dataset": False,
        "comments": "best"
    },
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": False,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": 2e-6,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "best"
    },
    {
        "input_feature_type": "explicit_and_single_img_input",  # explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input
        "use_object_visible_side": True,
        "use_rear_light_status": True,
        "model": "Embedding_CNN_LSTM",  # Embedding_Temporal_LSTM, Embedding_Transformer, CNN_LSTM, Embedding_CNN_LSTM, CNN_Transformer, TimeSformerNet, VideoMAENet
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
        "step_size": 20,
        "gamma": 0.01,
        "lr_cosine_t_max": 1.5,
        "lr_cosine_eta_min": 5e-7,
        "num_epochs": 30,
        "global_lr": True, 
        "optimizer": "SGD",  #SGD, Adam, AdamW
        "loss_function": "FocalLoss",
        "focal_loss_gamma": 2,
        "amp_enabled": True,
        "batch_size": 16,
        "freeze_strategy": None,  #head, partial, full
        "dropout_cnn_dynamic": 0.0,
        "dropout_cnn": 0.5,
        "dropout_pre_attention": 0.0,
        "dropout_fc": 0.5,
        "backbone_lr": 4e-4,
        "head_lr": 0.0,
        "weight_decay": 0.001,
        "sequence_stride": 4, 
        "cached_dataset": False,
        "comments": "best"
    },
]