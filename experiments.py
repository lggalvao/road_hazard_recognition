EXPERIMENTS = [
    {
        "exp_id": "A1",
        "input_feature_type": "explicit_feature",
        "object_visible_side": True,
        "tailight_status": True,
        "model": "Embedding_Temporal_LSTM",
        "input_img_type1": None,
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
    {
        "exp_id": "A1",
        "input_feature_type": "explicit_feature",
        "object_visible_side": False,
        "tailight_status": False,
        "model": "Embedding_Temporal_LSTM",
        "input_img_type1": None,
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
    {
        "exp_id": "A2",
        "input_feature_type": "single_img_input",
        "object_visible_side": None,
        "tailight_status": None,
        "model": "CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
    {
        "exp_id": "A3",
        "input_feature_type": "Two Images",
        "model": "MultiStream_CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": "img_local_context_ROI_0",
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
    {
        "exp_id": "A4",
        "input_feature_type": "Explicit + Single",
        "model": "Embedding_CNN_LSTM",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
    {
        "exp_id": "A5",
        "input_feature_type": "Single Image",
        "model": "VisionTransformer",
        "input_img_type1": "img_local_context_ROI_0",
        "input_img_type2": None,
        "enc_input_seq_length": 16,
        "classes_type": "literature_classes",
        "comments": None
    },
]
