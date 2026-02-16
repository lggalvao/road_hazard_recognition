#["FocalLoss", "weighted_FocalLoss", "weighted_CELoss", "CELoss"]

EXPERIMENTS_0 = [
    {
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 0,
        "loss_function": "CELoss",
        "comments": None
    },
    {
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 0,
        "loss_function": "CELoss",
        "comments": None
    },
    {
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 1,
        "loss_function": "FocalLoss",
        "comments": None
    },
    {
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 1,
        "loss_function": "FocalLoss",
        "comments": None
    },
    {
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_Transformer",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 1,
        "loss_function": "FocalLoss",
        "comments": None
    },
    # EXPLICIT FFEATURES
    #{
    #    "input_feature_type": "explicit_feature",
    #    "object_visible_side": False,
    #    "tailight_status": False,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    #{
    #    "input_feature_type": "explicit_feature",
    #    "object_visible_side": True,
    #    "tailight_status": False,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    #{
    #    "input_feature_type": "explicit_feature",
    #    "object_visible_side": False,
    #    "tailight_status": True,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    #{
    #    "input_feature_type": "explicit_feature",
    #    "object_visible_side": True,
    #    "tailight_status": True,
    #    "model": "Embedding_Temporal_LSTM",
    #    "input_img_type1": None,
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    ## SINGLE IMAGE
    #{
    #    "input_feature_type": "single_img_input",
    #    "object_visible_side": None,
    #    "tailight_status": None,
    #    "model": "CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_0",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    #{
    #    "input_feature_type": "single_img_input",
    #    "object_visible_side": None,
    #    "tailight_status": None,
    #    "model": "CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_0",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    ## MULTI IMAGE
    #{
    #    "input_feature_type": "Two Images",
    #    "model": "MultiStream_CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_0",
    #    "input_img_type2": "img_local_context_ROI_0",
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #},
    ##SINGLE IMAGE + EXPLICIT FEATURES
    #{
    #    "input_feature_type": "Explicit + Single",
    #    "model": "Embedding_CNN_LSTM",
    #    "input_img_type1": "img_local_context_ROI_0",
    #    "input_img_type2": None,
    #    "enc_input_seq_length": 16,
    #    "classes_type": "literature_classes",
    #    "stage": 0,
    #    "comments": None
    #}
]


EXPERIMENTS_1 = [
    {
        "input_feature_type": "explicit_and_single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "Embedding_CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 0,
        "loss_function": "CELoss",
        "comments": None
    },
    {
        "input_feature_type": "explicit_and_single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "Embedding_CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_1",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "stage": 0,
        "loss_function": "CELoss",
        "comments": None
    },
]