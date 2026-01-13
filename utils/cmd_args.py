import argparse




def get_cmd_args():
    
    parser = argparse.ArgumentParser("Main Arguments")
    
    parser.add_argument(
        "--device",
        help="Which device to use to run model.",
        type=int,
        default=0
    )
    parser.add_argument(
        "--input_feature_type",
        help="Choices: explicit_feature, single_img_input, multi_img_input, explicit_and_single_img_input, explicit_and_multi_img_input",
        type=str,
        default=None
    )
    parser.add_argument(
        "--input_img_type1",
        help="img_local_context, img_original, img_depth_info, img_local_depth_context, img_global_OOI_context, img_graph_rep, img_no_target_object, img_overlayed, img_with_lane_info, img_local_context_with_motion, img_local_context_ROI_0, img_local_context_ROI_3, img_local_context_ROI_5",
        type=str,
        default=None
    )
    parser.add_argument(
        "--input_img_type2",
        help="img_local_context, img_original, img_depth_info, img_local_depth_context, img_global_OOI_context, img_graph_rep, img_no_target_object, img_overlayed, img_with_lane_info, img_local_context_with_motion, img_local_context_ROI_0, img_local_context_ROI_3, img_local_context_ROI_5",
        type=str,
        default=None
    )
    parser.add_argument(
        "--input_img_type3",
        help="img_local_context, img_original, img_depth_info, img_local_depth_context, img_global_OOI_context, img_graph_rep, img_no_target_object, img_overlayed, img_with_lane_info, img_local_context_with_motion, img_local_context_ROI_0, img_local_context_ROI_3, img_local_context_ROI_5",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model",
        help="Choices: Embedding_LSTM, C3D, CNN_LSTM, VisionTransformer, Multi_Stream_CNN_LSTM, C3D_Embedding_LSTM, Embedding_CNN_LSTM, Embedding_Multi_Stream_CNN_LSTM, Spatial_CNN_LSTM, Spatial_Temporal_CNN_LSTM, CNN_TA_LSTM, multilayer_CNN_LSTM, Trajectory_Embedding_LSTM",
        type=str,
        default=None
    )
    parser.add_argument(
        "--classes_type",
        help="Choices: motion_towards, all_classes, literature_classes, 20_or_more",
        type=str,
        default=None
    )
    
    return parser.parse_args()