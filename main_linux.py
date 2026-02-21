#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


from __future__ import print_function
if __name__ == '__main__':
    import os
    import torch
    from data.dataset import (
        create_or_load_dataset,
        split_roadHazardDataset,
        RoadHazardDataset,
    )
    from models.load_models_parameters import (
        load_model,
        select_config_setting,
        unfreeze_layer4
    )
    from utils.seeds import setup_seed
    from utils.logger import (
        record_parameters_and_results
    )
    from torch.utils.data import DataLoader
    import time
    import numpy as np
    from sklearn.metrics import (
        recall_score,
        accuracy_score,
        f1_score,
        precision_score,
        confusion_matrix,
        ConfusionMatrixDisplay,
        classification_report,
        precision_recall_fscore_support
    )
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import pandas as pd
    import cv2
    from pathlib import Path
    from early_stopping_pytorch import EarlyStopping
    import wandb
    #from tensorflow.keras.callbacks import ReduceLROnPlateau
    from train.trainer import train_model, estimate_training_time
    from train.tester import test_model
    from train.losses import get_loss_function
    from utils.config import Config, load_config, build_paths
    from utils.check_point import save_config
    from utils.logger import log_config_to_wandb, initialize_wandb, setup_logging
    from train.optimizer import get_optimizer
    from utils.cmd_args import get_cmd_args
    from utils.setup_hardware import get_devie
    from utils.timing import timeit
    import logging
    from experiments import (
        EXPERIMENTS_0,
        EXPERIMENTS_1
    )
    from types import SimpleNamespace
    
    import torch

    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    logger = setup_logging("./output/training.log", level=logging.INFO)
    logger = logging.getLogger("hazard_recognition")
    
    config_file_path = "C:/Projects/hazard_prediction_project/output/trained_models/explicit_feature_5/config.json"

    cfg = Config()
    #cfg = load_config(config_file_path)
    cfg = build_paths(cfg)
    cmd_args = get_cmd_args()

    visible_side_arr = ['front_side', 'front_left_side', 'front_right_side', 'rear_side', 'rear_right_side', 'rear_left_side', 'left_side', 'right_side', 'UNK']
    tailight_status_arr = ['BOO', 'OLO', 'OLR', 'OOO', 'OOR', 'BLO', 'BLR', 'BOR', 'REVERSE', 'UNK']

    cfg.system.multi_gpu = False
    cfg.data.with_no_hazard_samples_flag = True
    cfg.data.sequence_stride = 1
    cfg.data.dataset_trim = 500
    cfg.model.freeze_strategy = "head"
    cfg.training.ts_augme = False
    cfg.loss.focal_loss_gamma = 1
    cfg.training.patience = 5
    cfg.data.split_dataset = True
    cfg.logging.pred_save_wrong = False
    cfg.logging.pred_save_frame_flag = False
    cfg.logging.test_name = 'explicit_feature_5'

    get_devie(cfg, cmd_args)

    if cmd_args.device == 0:
        exp = EXPERIMENTS_0
    
    elif cmd_args.device == 1:
        exp = EXPERIMENTS_1
    
    for exp_config in exp:

        # Enable to use dot .object
        exp_config = SimpleNamespace(**exp_config)
        cfg.data.input_feature_type = exp_config.input_feature_type
        cfg.data.input_img_type1 = exp_config.input_img_type1
        cfg.data.input_img_type2 = exp_config.input_img_type2
        cfg.model.model = exp_config.model
        cfg.model.classes_type = exp_config.classes_type
        cfg.loss.loss_function = exp_config.loss_function

        train_flag = 'train' # Run 'train', 'test', or 'prediction' algorithm.
        if train_flag == 'test' or train_flag == 'prediction':

            #log_file_path =  './output/trained_models/official_results_v2/' + 'explicit_feature' + '/'+ cfg.logging.test_name
            log_file_path = './output/trained_models/' + cfg.logging.test_name
            config_file_path = log_file_path + "/config.json"
            load_config(config_file_path)
            print('log_file_path', log_file_path)

        select_config_setting(cfg)
        cfg.model.object_visible_side = exp_config.object_visible_side
        cfg.model.tailight_status = exp_config.tailight_status
        cfg.model.enc_input_seq_length = exp_config.enc_input_seq_length
        cfg.logging.comments = exp_config.comments
        cfg.training.stage = exp_config.stage
        cfg.training.batch_size = exp_config.batch_size
        cfg.data.cached_dataset = exp_config.cached_dataset
        
        #results_csv = pd.read_csv(cfg.logging.results_csv_file_path)
        
        setup_seed(cfg.system.seed)

        allsetDataloader = create_or_load_dataset(cfg)
        
        logger.info("Loading Model")
        net = load_model(cfg, allsetDataloader)
        logger.info(f" Making sure model is using GPU: {next(net.parameters()).device}")
        
        if train_flag == 'train':

            log_file_path, file_number = record_parameters_and_results(cfg)
            
            if cfg.training.stage == 0:
                optimizer = get_optimizer(cfg, net)

                exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=cfg.training.step_size,
                    gamma=cfg.training.gamma
                )
    
                criterion = get_loss_function(cfg)
                
                early_stopping = EarlyStopping(
                    patience=cfg.training.patience,
                    verbose=True,
                    path=log_file_path + f'/stage{cfg.training.stage}_validation_best.tar'
                )
    
                run_wandb = initialize_wandb(
                    cfg,
                    log_file_path
                )
    
                log_config_to_wandb(cfg)
                
                save_config(cfg, log_file_path + "/config.json")
                
                train_model(
                    cfg,
                    net,
                    allsetDataloader,
                    optimizer,
                    exp_lr_scheduler,
                    criterion,
                    early_stopping,
                    run_wandb,
                    log_file_path
                )
                
            if cfg.training.stage == 1:
                optimizer = get_optimizer(cfg, net)
    
                exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=cfg.training.step_size,
                    gamma=cfg.training.gamma
                )
    
                criterion = get_loss_function(cfg)
    
                early_stopping = EarlyStopping(
                    patience=cfg.training.patience,
                    verbose=True,
                    path=log_file_path + '/validation_best.tar'
                )
    
                run_wandb = initialize_wandb(
                    cfg,
                    log_file_path
                )
    
                log_config_to_wandb(cfg)
                
                train_model(
                    cfg,
                    net,
                    allsetDataloader,
                    optimizer,
                    exp_lr_scheduler,
                    criterion,
                    early_stopping,
                    run_wandb,
                    log_file_path
                )
                
                cfg.training.stage = 2
                
                net.load_state_dict(torch.load(log_file_path + "/validation_best.tar", map_location=torch.device(cfg.system.device)))
                
                unfreeze_layer4(net)
                
                optimizer = get_optimizer(cfg, net)
                
                exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=cfg.training.step_size,
                    gamma=cfg.training.gamma
                )
                
                early_stopping = EarlyStopping(
                    patience=cfg.training.patience,
                    verbose=True,
                    path=log_file_path + '/validation_best_statge_2.tar'
                )
                
                run_wandb.finish()
                
                run_wandb = initialize_wandb(
                    cfg,
                    log_file_path
                )
    
                log_config_to_wandb(cfg)
                
                train_model(
                    cfg,
                    net,
                    allsetDataloader,
                    optimizer,
                    exp_lr_scheduler,
                    criterion,
                    early_stopping,
                    run_wandb,
                    log_file_path
                )
            
            test_model(
                cfg,
                net,
                allsetDataloader,
                run_wandb,
                log_file_path
            )

            run_wandb.finish()

        if train_flag == 'test':
            run_wandb = None
            test_model(
                cfg,
                net,
                allsetDataloader,
                run_wandb,
                log_file_path
            )

        if train_flag == 'prediction':
# ##        ###############################################################################################
                                    # # GENERATING PREDICTION RESULTS
# ##        ###############################################################################################
                    print('\n Running Prediction Evaluation\n')

                    # Clear GPU memory
                    torch.cuda.empty_cache()

                    best_paths = []
                    load_file = 'None'

                    best_1 = Path(log_file_path + '/best_1.tar')
                    best_2 = Path(log_file_path + '/best_2.tar')
                    best_last = Path(log_file_path + '/best_last.tar')
                    validation_best = Path(log_file_path + '/validation_best.tar')

                    if best_1.exists():
                        load_file = best_1
                    elif best_2.exists():
                        load_file = best_2
                    elif best_last.exists():
                        load_file = best_last
                    elif validation_best.exists():
                        load_file = validation_best

                    print(load_file)
                    net.load_state_dict(torch.load(load_file, map_location=torch.device(cfg.system.device)))

                    net.eval()

                    ## Note: It is required to re-load the test dataset to put the videos in sequence
                    cfg.training.batch_size = 840
                    tsSet = RoadHazardDataset(cfg, test_output_dir, phase = 'test')
                    tsDataloader = DataLoader(tsSet,batch_size= cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers, drop_last = True)

                    print("Test Sequence Samples:",len(tsSet))
                    print("Number of test batches:",len(tsDataloader))

                    # Variables to store all the batches data.
                    test_lat_preds_np = []
                    test_lat_trues_np = []
                    all_together_batches = []
                    lat_true_all_batches = []
                    video_nu_all_batches = []
                    dir_path_all_batches = []
                    frame_nu_all_batches = []
                    f_0_all_batches = []
                    f_1_all_batches = []
                    f_2_all_batches = []
                    labeled_img_dir_path_all_batches = []
                    hazard_name_all_batches = []

                    for i, data in enumerate(tqdm(tsDataloader, desc="Predicting...")):
                        st_time = time.time()

                        if cfg.data.input_feature_type == 'explicit_feature':
                            all_together_norm, true_hazard_event, video_nu, frame_nu, _, start_frame, end_frame, hazard_name_hist, all_together_hist_info, img_path_root_hist = data
                            all_together_norm = all_together_norm.to(cfg.system.device)

                        elif cfg.data.input_feature_type == 'single_img_input' or cfg.data.input_feature_type == 'explicit_and_single_img_input':
                            img_vector, all_together_norm, true_hazard_event, video_nu, frame_nu, _, start_frame, end_frame, hazard_name_hist, all_together_hist_info, img_path_root_hist= data
                            img_vector = img_vector.to(cfg.system.device)
                            all_together_norm = all_together_norm.to(cfg.system.device)

                        elif cfg.data.input_feature_type == 'multi_img_input' or cfg.data.input_feature_type == 'explicit_and_multi_img_input':
                            img_vector, img_vector_2, img_vector_3, all_together_norm, true_hazard_event, video_nu, frame_nu, _, start_frame, end_frame, hazard_name_hist, all_together_hist_info, img_path_root_hist= data
                            img_vector = img_vector.to(cfg.system.device)
                            img_vector_2 = img_vector_2.to(cfg.system.device)
                            img_vector_3 = img_vector_3.to(cfg.system.device)
                            all_together_norm = all_together_norm.to(cfg.system.device)

                        true_hazard_event = true_hazard_event.to(cfg.system.device)

                        if cfg.data.input_feature_type == 'explicit_feature':
                            pred_hazard_event = net(all_together_norm)

                        elif cfg.data.input_feature_type == 'single_img_input':
                            pred_hazard_event = net(img_vector)

                        elif cfg.data.input_feature_type == 'multi_img_input' and cfg.data.num_of_input_img == 2:
                            pred_hazard_event = net(img_vector, img_vector_2)

                        elif cfg.data.input_feature_type == 'multi_img_input' and cfg.data.num_of_input_img == 3:
                            pred_hazard_event = net(img_vector, img_vector_2, img_vector_3)

                        elif cfg.data.input_feature_type == 'explicit_and_single_img_input':
                            pred_hazard_event = net(all_together_norm, img_vector)

                        elif cfg.data.input_feature_type == 'multi_img_input' and cfg.data.num_of_input_img == 2:
                            pred_hazard_event = net(all_together_norm, img_vector, img_vector_2)

                        elif cfg.data.input_feature_type == 'multi_img_input' and cfg.data.num_of_input_img == 3:
                            pred_hazard_event = net(all_together_norm, img_vector, img_vector_2, img_vector_3)

                        true_hazard_event = true_hazard_event.float()
                        test_lat_pred_np = pred_hazard_event.detach().cpu().numpy()
                        test_lat_true_np = true_hazard_event.detach().cpu().numpy()
                        test_lat_pred_np = np.argmax(test_lat_pred_np, 1)
                        test_lat_true_np = np.argmax(test_lat_true_np, 1)

                        test_lat_preds_np.extend(test_lat_pred_np)
                        test_lat_trues_np.extend(test_lat_true_np)

                        all_together_batches.extend(all_together_hist_info.detach().cpu().numpy())
                        lat_true_all_batches.extend(true_hazard_event.detach().cpu().numpy())
                        video_nu_all_batches.extend(video_nu)
                        frame_nu_all_batches.extend(frame_nu.detach().cpu().numpy())
                        f_0_all_batches.extend(start_frame)
                        f_1_all_batches.extend(end_frame)
                        hazard_name_all_batches.extend(hazard_name_hist)
                        dir_path_all_batches.extend(img_path_root_hist)

                        #for i in range(all_together_hist_info.shape[0]):
                        #    all_together_all = all_together_hist_info[i]
                        #    temp_frame_all = frame_nu[i].detach().cpu().numpy()
                        #    for y in range(len(all_together_all[0])):
                        #
                        #        print('img_dir_all[y]:', img_path_root_hist[y])
                        #        print('temp_frame_all[y]:', temp_frame_all[y])
                        #        image_path = img_path_root_hist[i] + str(temp_frame_all[y]).zfill(5) + '.png'
                        #        frame = cv2.imread(image_path)
                        #        frame = cv2.resize(frame, (new_img_width, new_img_height), interpolation = cv2.INTER_AREA)
                        #
                        #        #['object_type', 'tailight_status_int', 'object_visible_side_int', 'xc', 'yc', 'w', 'h', 'xc_speed', 'yc_speed', 'x_1', 'y_1', 'x_2', 'y_2']
                        #        x_1, y_1, x_2, y_2 = int(all_together_all[y][9]), int(all_together_all[y][10]), int(all_together_all[y][11]), int(all_together_all[y][12])
                        #        xc, yc = int(all_together_all[y][3]), int(all_together_all[y][4])
                        #        tailight_status = int(all_together_all[y][1])
                        #        object_visible_side = int(all_together_all[y][2])
                        #
                        #        PR_int = "None"
                        #        PR_int = cfg.model.classes_name[int(test_lat_pred_np[i])]
                        #
                        #        GT_int = "None"
                        #        GT_int = cfg.model.classes_name[int(test_lat_true_np[i])]
                        #
                        #        font = cv2.FONT_HERSHEY_PLAIN
                        #        #BGR Format
                        #        cv2.putText(frame, 'Video_nu: '             + str(video_nu[y]),   (5   , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'Curr_Frame: '           + str(temp_frame_all[y]), (150 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'f_0: '                  + str(start_frame[y]),         (325 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'f_1: '                  + str(end_frame[y]),         (475 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'PR_int: '               + PR_int + '|' + str(test_lat_pred_np[i]),                 (600 , 15)        , font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'GT_int: '               + GT_int + '|' + str(test_lat_true_np[i]),                 (850 , 15)        , font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'T_S: '      + tailight_status_arr[tailight_status],        (int(xc), int(yc)-30), font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        #        cv2.putText(frame, 'O_V_S: '  + visible_side_arr[object_visible_side],    (int(xc), int(yc)-15), font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        #        cv2.circle(frame, (int(xc), int(yc)), radius=5, color=(0, 255, 255), thickness=-1)
                        #        frame = cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
                        #        'tailight_status', 'object_visible_side'
                        #
                        #        cv2.imshow('frame',frame)
                        #        cv2.waitKey(10)



                    test_lat_trues_np = np.array(test_lat_trues_np)
                    test_lat_preds_np = np.array(test_lat_preds_np)
                    indices_to_remove = np.where(test_lat_trues_np == 10.0)[0]
                    test_lat_trues_np = np.delete(test_lat_trues_np, indices_to_remove)
                    test_lat_preds_np = np.delete(test_lat_preds_np, indices_to_remove)

                    test_lat_acc = accuracy_score(test_lat_trues_np, test_lat_preds_np)
                    print('test_lat_acc: ', test_lat_acc)
                    print('classification_report: ', classification_report(test_lat_trues_np, test_lat_preds_np))

                    exit()

                    if cfg.logging.pred_save_wrong:
                        save_frames_dir = log_file_path + "/frame_folder_wrong/"
                    else:
                        save_frames_dir = log_file_path + "/frame_folder_all/"

                    try:
                        os.mkdir(save_frames_dir)
                    except OSError:
                        print ("Creation of the directory failed" )

                    # Change the current directory
                    os.chdir(save_frames_dir)

                    temp_frame =  []
                    all_together_hist_n =  []
                    lat_true_n =  []
                    lat_pred_n =  []
                    img_dir_n =  []
                    video_nu_n = []
                    f0_n = []
                    f1_n = []

                    temp_frame_all = []
                    all_together_all = []
                    lat_true_all = []
                    lat_pred_all = []
                    img_dir_all = []
                    video_nu_all = []
                    f0_all = []
                    f1_all = []

                    video = -1

                    for x in range (len(video_nu_all_batches)):

                        if (video != video_nu_all_batches[x]):
                            temp_frame_all.extend(temp_frame)
                            all_together_all.extend(all_together_hist_n)
                            lat_true_all.extend(lat_true_n)
                            lat_pred_all.extend(lat_pred_n)
                            img_dir_all.extend(img_dir_n)
                            video_nu_all.extend(video_nu_n)
                            f0_all.extend(f0_n)
                            f1_all.extend(f1_n)

                            video = video_nu_all_batches[x] # Assign the current clip number
                            video_nu_n = video_nu_all_batches[x] # Put the first value to avoid error
                            for i in range(cfg.model.enc_input_seq_length-1):
                                video_nu_n = np.hstack([video_nu_n, video_nu_all_batches[x]])

                            img_dir_n = dir_path_all_batches[x]
                            for i in range(cfg.model.enc_input_seq_length-1):
                                img_dir_n = np.hstack([img_dir_n, dir_path_all_batches[x]])


                            lat_true_n = test_lat_trues_np[x]
                            for i in range(cfg.model.enc_input_seq_length-1):
                                lat_true_n = np.hstack([lat_true_n, test_lat_trues_np[x]])

                            lat_pred_n = test_lat_preds_np[x]
                            for i in range(cfg.model.enc_input_seq_length-1):
                                lat_pred_n = np.hstack([lat_pred_n, test_lat_preds_np[x]])

                            temp_frame = frame_nu_all_batches[x]
                            all_together_hist_n = all_together_batches[x]

                            f0_n = f_0_all_batches[x]
                            for i in range(cfg.model.enc_input_seq_length-1):
                                f0_n = np.hstack([f0_n, f_0_all_batches[x]])
                            f1_n = f_1_all_batches[x]
                            for i in range(cfg.model.enc_input_seq_length-1):
                                f1_n = np.hstack([f1_n, f_1_all_batches[x]])

                        else:

                            temp_frame = np.hstack([temp_frame, frame_nu_all_batches[x][-1]])
                            all_together_hist_n = np.vstack([all_together_hist_n, all_together_batches[x][-1]])
                            lat_true_n = np.hstack([lat_true_n, test_lat_trues_np[x]])
                            lat_pred_n = np.hstack([lat_pred_n, test_lat_preds_np[x]])
                            video_nu_n = np.hstack([video_nu_n, video_nu_all_batches[x]])
                            f0_n = np.hstack([f0_n, f_0_all_batches[x]])
                            f1_n = np.hstack([f1_n, f_1_all_batches[x]])
                            img_dir_n = np.hstack([img_dir_n, dir_path_all_batches[x]])

                    video_number = -1
                    pred_horiz_all = np.empty([1, 4])
                    TTE_index = 0
                    F0_index = 0

                    F_to_LCE_GT_all    = []
                    F_to_LCE_PRED_all  = []
                    F0_to_LCE_GT_all   = []
                    F0_to_LCE_PRED_all = []

                    for y in range(len(all_together_all)):

                        print('img_dir_all[y]:', img_dir_all[y])
                        print('temp_frame_all[y]:', temp_frame_all[y])
                        image_path = img_dir_all[y] + str(temp_frame_all[y]).zfill(5) + '.png'
                        frame = cv2.imread(image_path)
                        frame = cv2.resize(frame, (new_img_width, new_img_height), interpolation = cv2.INTER_AREA)

                        #['object_type', 'tailight_status_int', 'object_visible_side_int', 'xc', 'yc', 'w', 'h', 'xc_speed', 'yc_speed', 'x_1', 'y_1', 'x_2', 'y_2']
                        x_1, y_1, x_2, y_2 = int(all_together_all[y][9]), int(all_together_all[y][10]), int(all_together_all[y][11]), int(all_together_all[y][12])
                        xc, yc = int(all_together_all[y][3]), int(all_together_all[y][4])
                        tailight_status = int(all_together_all[y][1])
                        object_visible_side = int(all_together_all[y][2])

                        PR_int = "None"
                        PR_int = cfg.model.classes_name[int(lat_pred_all[y])]

                        GT_int = "None"
                        GT_int = cfg.model.classes_name[int(lat_true_all[y])]

                        font = cv2.FONT_HERSHEY_PLAIN
                        #BGR Format
                        cv2.putText(frame, 'Video_nu: '             + str(video_nu_all[y]),   (5   , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        cv2.putText(frame, 'Curr_Frame: '           + str(temp_frame_all[y]), (150 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        cv2.putText(frame, 'f_0: '                  + str(f0_all[y]),         (325 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        cv2.putText(frame, 'f_1: '                  + str(f1_all[y]),         (475 , 15)        , font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
                        cv2.putText(frame, 'PR_int: '               + PR_int + '|' + str(lat_pred_all[y]),                 (600 , 15)        , font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        cv2.putText(frame, 'GT_int: '               + GT_int + '|' + str(lat_true_all[y]),                 (850 , 15)        , font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        cv2.putText(frame, 'T_S: '      + tailight_status_arr[tailight_status],        (int(xc), int(yc)-30), font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        cv2.putText(frame, 'O_V_S: '  + visible_side_arr[object_visible_side],    (int(xc), int(yc)-15), font, 1, (0  , 110, 255), 2, cv2.LINE_4)
                        cv2.circle(frame, (int(xc), int(yc)), radius=5, color=(0, 255, 255), thickness=-1)
                        frame = cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
                        'tailight_status', 'object_visible_side'
