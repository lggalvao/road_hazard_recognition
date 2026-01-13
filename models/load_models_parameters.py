import torch
import torch.nn as nn
from models import (
    Embedding_Temporal_LSTM,
    Embedding_Transformer,
    CNN_LSTM
)
from utils.timing import timeit

@timeit
def load_model(cfg, allsetDataloader):
    """
    Build and initialize a model based on cfg settings
    using a single sample batch to infer input shapes.
    """

    # ----------------------------------------------------------------------
    # 1. Pull ONE batch from dataloader to infer shapes
    # ----------------------------------------------------------------------
    batch = next(iter(allsetDataloader["train"]))

    # ----------------------------------------------------------------------
    # 2. Extract feature shapes based on input_feature_type
    # ----------------------------------------------------------------------
    feature_type = cfg.data.input_feature_type

    # Always available for every mode
    #features_norm = batch["features_norm"]
    kinematic = batch["kinematic"]
    bbox = batch["bbox"]

    cfg.model.num_kinematic_features = kinematic.shape[-1]
    cfg.model.num_bbox_features = bbox.shape[-1]
    #cfg.model.num_kinematic_features = kinematic.shape[-1] + bbox.shape[-1]
    
    #object_type = batch["object_type"]
    #object_visible_side = batch["object_visible_side"]
    #tailight_status = batch["tailight_status"]

    # Image inputs vary depending on mode
    if feature_type in ["single_img_input", "explicit_and_single_img_input"]:
        # one image stream
        img_tensor = batch["images"][0]
        cfg.data.img_shape = img_tensor.shape

    elif feature_type in ["multi_img_input", "explicit_and_multi_img_input"]:
        # multiple image streams
        img_streams = batch["images"]
        img_tensor = torch.stack(img_streams[0], dim=0)
        cfg.data.img_shape = img_tensor.shape

    # explicit_feature mode does not need images
    # trajectory mode handled separately below

    # ----------------------------------------------------------------------
    # 3. Instantiate model according to configuration
    # ----------------------------------------------------------------------
    model_name = cfg.model.model

    if feature_type == "explicit_feature":
        if model_name == "Embedding_Transformer":
            net = Embedding_Transformer.Embedding_Transformer(cfg)

        elif model_name == "Embedding_Temporal_LSTM":
            net = Embedding_Temporal_LSTM.Embedding_Temporal_LSTM(cfg)

        else:
            raise ValueError(f"Unsupported model {model_name} for explicit features")

    elif feature_type == "single_img_input":

        if model_name == "CNN_LSTM":
            net = CNN_LSTM.CNN_LSTM(cfg)
            if cfg.training.stage == 1:
                freeze_cnn(net)

        else:
            raise ValueError(f"Unsupported model {model_name}")

    else:
        raise ValueError(f"Unsupported input_feature_type {feature_type}")

    # ----------------------------------------------------------------------
    # 4. Multi-GPU support
    # ----------------------------------------------------------------------
    if cfg.system.multi_gpu:
        net = nn.DataParallel(net, device_ids=[0, 1])

    # ----------------------------------------------------------------------
    # 5. Move to device
    # ----------------------------------------------------------------------
    net = net.to(cfg.system.device)
    return net


def Trajectory_Embedding_LSTM_parameters(cfg):
    cfg.system.seed = 100
    cfg.data.num_of_input_img = None
    cfg.data.input_img_resize = None
    
    #Embedding parameters
    cfg.model.object_visible_side = True
    cfg.model.tailight_status = True
    cfg.model.output_embedding_size = 112
    cfg.model.dropout_embedding_feature = 0.25
    
    #LSTM parameters
    cfg.model.enc_hidden_size = cfg.model.output_embedding_size
    cfg.model.enc_input_seq_length = 13 #25 frames per second, hence 25 for a observation horizon of 1
    cfg.model.encoder_type = 'LSTM' #LSTM, GRU
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False

    #FC Parameters
    cfg.model.fc_output_size = 128
    cfg.model.dropout1 = 0.8
    cfg.training.clip_grad = 5

    #Hyperparameters
    cfg.training.batch_size = 128
    cfg.training.num_epochs = 40#15
    cfg.training.optimizer = 'Adam' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0001#Learnign rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0006
    cfg.training.step_size = 25
    cfg.training.gamma = 0.01
    cfg.logging.comments = 'none'


def select_config_setting(cfg):
    
    if cfg.data.input_feature_type == 'explicit_feature':
        explicit_feature_parameters(cfg)
        cfg.logging.file_name = 'explicit_feature'
        cfg.logging.results_csv_file_path = "./Excel Results/results_explicit.csv"
    
    elif cfg.data.input_feature_type == 'single_img_input':
        single_img_input_parameters(cfg)
        cfg.logging.file_name = 'single_img_input'
        cfg.logging.results_csv_file_path = "./Excel Results/results_single.csv"
    
    elif cfg.data.input_feature_type == 'multi_img_input':
        multi_img_input_parameters(cfg)
        cfg.logging.file_name = 'multi_img_input'
        cfg.logging.results_csv_file_path = "./Excel Results/results_multi.csv"
    
    elif cfg.data.input_feature_type == 'explicit_and_single_img_input':
        explicit_and_single_img_input_parameters(cfg)
        cfg.logging.file_name = 'explicit_and_single_img_input'
        cfg.logging.results_csv_file_path = "./Excel Results/results_single_explicit.csv"
    
    elif cfg.data.input_feature_type == 'explicit_and_multi_img_input':
        explicit_and_multi_img_input_parameters(cfg)
        cfg.logging.file_name = 'explicit_and_multi_img_input'
        cfg.logging.results_csv_file_path = "./Excel Results/results_multi_explicit.csv"
    
    elif cfg.model.model == 'Trajectory_Embedding_LSTM':
        Trajectory_Embedding_LSTM_parameters(cfg)
        cfg.logging.file_name = 'Trajectory_Embedding_LSTM'


def explicit_feature_parameters(cfg):
    cfg.system.seed = 100
    cfg.data.num_of_input_img = None
    cfg.data.input_img_resize = None
    cfg.data.num_of_no_hazard_samples_train = 250
    cfg.data.num_of_no_hazard_samples_test = 50
    
    #Embedding parameters
    cfg.model.object_visible_side = True
    cfg.model.tailight_status = True
    cfg.model.output_embedding_size = 112
    cfg.model.dropout_embedding_feature = 0.25
    
    #LSTM parameters
    cfg.model.enc_hidden_size = cfg.model.output_embedding_size
    cfg.model.enc_input_seq_length = 13 #25 frames per second, hence 25 for a observation horizon of 1s
    cfg.model.encoder_type = 'LSTM' #LSTM, GRU
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False

    
    #FC Parameters
    cfg.model.fc_output_size = 128
    cfg.model.dropout1 = 0.7
    cfg.training.clip_grad = 5

    #Hyperparameters
    cfg.training.batch_size = 128
    cfg.training.num_epochs = 35#30 15
    cfg.training.optimizer = 'Adam' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0001#Learnign rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0009
    cfg.training.step_size = 16
    cfg.training.gamma = 0.1
    cfg.logging.comments = 'none'


def single_img_input_parameters(cfg):
    print('Using single_img_input_parameters')
    cfg.system.seed = 205
    cfg.data.num_of_input_img = None
    cfg.data.num_of_no_hazard_samples_train = 250
    cfg.data.num_of_no_hazard_samples_test = 50
    
    #CNN parameters
    cfg.data.input_img_resize = (224, 224) # (h, w) (128, 171) (112, 112)
    
    #Embedding parameters
    cfg.model.object_visible_side = False
    cfg.model.tailight_status = False
    cfg.model.output_embedding_size = None
    
    #LSTM parameters
    cfg.model.enc_hidden_size = 128
    cfg.model.enc_input_seq_length = 13 #10 fps hence 1.6s
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False
    
    #Attention
    cfg.model.spatial_attention_channels = None
    
    #Output parameters
    cfg.model.fc_output_size = None
    cfg.model.dropout1 = 0.8
    cfg.training.clip_grad = 5
    
    #Hyperparameters
    cfg.training.batch_size = 32 #CNN:50, Vision Transformer:20
    cfg.training.num_epochs = 30
    cfg.training.optimizer = 'SGD' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0001 #0.00009#Learning rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0001 #0.0001
    cfg.training.step_size = 40
    cfg.training.gamma = 0.01
    cfg.logging.comments = None


def multi_img_input_parameters(cfg):
    print('Using local_context_parameters')
    cfg.system.seed = 205
    cfg.data.num_of_input_img = 2
    cfg.data.num_of_no_hazard_samples_train = 250
    cfg.data.num_of_no_hazard_samples_test = 50
    
    #CNN parameters
    cfg.data.input_img_resize = (224, 224)
    
    #Embedding parameters
    cfg.model.object_visible_side = True
    cfg.model.tailight_status = False
    cfg.model.output_embedding_size = 'Not in use'
    
    #LSTM parameters
    cfg.model.enc_hidden_size = 128
    cfg.model.enc_input_seq_length = 18 #10 fps hence 1.6s
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False
    
    #Output parameters
    cfg.model.fc_output_size = 'Not in use'
    cfg.model.dropout1 = 0.8
    cfg.training.clip_grad = 5
    
    #Hyper-parameters
    cfg.training.batch_size = 50
    cfg.training.num_epochs = 13
    cfg.training.optimizer = 'SGD' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0003#Learning rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0001 #0.0001
    cfg.training.step_size = 4
    cfg.training.gamma = 0.33
    cfg.logging.comments = 'none'


def explicit_and_single_img_input_parameters(cfg):
    cfg.system.seed = 100
    cfg.data.num_of_input_img = 'Not in use'
    cfg.data.num_of_no_hazard_samples_train = 400
    cfg.data.num_of_no_hazard_samples_test = 100
    
    #CNN parameters
    cfg.data.input_img_resize = (224,224)
    
    #Embedding parameters
    cfg.model.object_visible_side = True
    cfg.model.tailight_status = True
    cfg.model.output_embedding_size = 576
    cfg.model.dropout_embedding_feature = 0.1
    
    #LSTM parameters
    cfg.model.enc_hidden_size = 128
    cfg.model.enc_input_seq_length = 13 #25 frames per second, hence 25 for a observation horizon of 1s.
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False
    
    #FC Parameters
    cfg.model.fc_output_size = 'Not in use'
    cfg.model.dropout1 = 0.8
    cfg.training.clip_grad = 5
    
    #Hyperparameters
    cfg.training.batch_size = 50
    cfg.training.num_epochs = 40
    cfg.training.optimizer = 'SGD' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0002#Learnign rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0001 #0.0001
    cfg.training.step_size = 9
    cfg.training.gamma = 0.01
    cfg.logging.comments = 'None'


def explicit_and_multi_img_input_parameters(cfg):
    cfg.system.seed = 205
    cfg.data.num_of_input_img = 2
    cfg.data.num_of_no_hazard_samples_train = 250
    cfg.data.num_of_no_hazard_samples_test = 50

    #CNN parameters
    cfg.data.input_img_resize = (224,224) # (h, w) (128, 171) (112, 112)
    
    #Embedding parameters
    cfg.model.object_visible_side = True
    cfg.model.tailight_status = True
    cfg.model.output_embedding_size = 576
    cfg.model.dropout_embedding_feature = 0.1
    
    #LSTM parameters
    cfg.model.enc_hidden_size = 128
    cfg.model.enc_input_seq_length = 13 #25 frames per second, hence 25 for a observation horizon of 1s.
    cfg.model.enc_layers_num = 1
    cfg.model.bi_directional = False
    
    #FC Parameters
    cfg.model.fc_output_size = 128
    cfg.model.dropout1 = 0.8
    cfg.training.clip_grad = 5
    
    #Hyperparameters
    cfg.training.batch_size = 50
    cfg.training.num_epochs = 15#25
    cfg.training.optimizer = 'SGD' #SGD, Adam, AdamW
    cfg.training.learning_rate = 0.0003 #Learnign rate for SGD(0.09), Adam(0.00006) using images
    cfg.training.cnn_lr = 1e-5
    cfg.training.weight_decay = 0.0001
    cfg.training.step_size = 15
    cfg.training.gamma = 0.1
    cfg.logging.comments = 'none'


def set_parameters(cfg):
    
    if cfg.data.input_feature_type == 'explicit_feature':
        explicit_feature_parameters(cfg)
        cfg.logging.file_name = 'explicit_feature'
    
    elif cfg.data.input_feature_type == 'single_img_input':
        single_img_input_parameters(cfg)
        cfg.logging.file_name = 'single_img_input'
    
    elif cfg.data.input_feature_type == 'multi_img_input':
        multi_img_input_parameters(cfg)
        cfg.logging.file_name = 'multi_img_input'
    
    elif cfg.data.input_feature_type == 'explicit_and_single_img_input':
        explicit_and_single_img_input_parameters(cfg)
        cfg.logging.file_name = 'explicit_and_single_img_input'
    
    elif cfg.data.input_feature_type == 'explicit_and_multi_img_input':
        explicit_and_multi_img_input_parameters(cfg)
        cfg.logging.file_name = 'explicit_and_multi_img_input'
    
    elif cfg.model.model == 'Trajectory_Embedding_LSTM':
        Trajectory_Embedding_LSTM_parameters(cfg)
        cfg.logging.file_name = 'Trajectory_Embedding_LSTM'


def freeze_cnn(net):
    for p in net.resnet.parameters():
        p.requires_grad = False

def unfreeze_layer4(net):
    for p in net.resnet[-1].parameters():
        p.requires_grad = True