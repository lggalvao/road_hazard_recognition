import cv2
import matplotlib.pyplot as plt
from pathlib import Path


new_img_height, new_img_width = 640, 1280

def debug_observation_sequence(all_together_hist, original_frame_path_hist, video_nu, frame_n, start_frame_hist, end_frame_hist, hazard_name_hist):
    print("all_together_hist", all_together_hist)
    all_together_all = all_together_hist
    print("all_together_all", all_together_all)
    img_dir_all = original_frame_path_hist
    
    video_nu_all = video_nu
    temp_frame_all = frame_n
    f0_all = start_frame_hist
    f1_all = end_frame_hist
    lat_true_all = hazard_name_hist
    print("hazard_name_hist", hazard_name_hist)
    for y in range(len(all_together_all)):
        print("y", y)
    
        image_path = img_dir_all[y]
        print('image_path;', image_path)
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (new_img_width, new_img_height), interpolation = cv2.INTER_AREA)
    
        #'object_type':0, 'tailight_status_int':1, 'object_visible_side_int':2, 'xc':3, 'yc':4, 'w':5, 'h':6, 'x_1':7, 'y_1':8, 'x_2':9, 'y_2:10'
        x_1, y_1, x_2, y_2 = int(all_together_all[y][7]), int(all_together_all[y][8]), int(all_together_all[y][9]), int(all_together_all[y][10])
        xc, yc = int(all_together_all[y][3]), int(all_together_all[y][4])
        
        GT_int = "None"
        GT_int = lat_true_all[0]
        
        font = cv2.FONT_HERSHEY_PLAIN        
        #BGR Format
        cv2.putText(frame, 'Video_nu: '   + str(video_nu_all),   (5   , 15), font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
        cv2.putText(frame, 'Curr_Frame: ' + str(temp_frame_all[y]), (150 , 15), font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
        cv2.putText(frame, 'f_0: '        + str(f0_all[0]),         (325 , 15), font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
        cv2.putText(frame, 'f_1: '        + str(f1_all[0]),         (475 , 15), font, 1, (240, 255, 0  ), 2, cv2.LINE_4)
        cv2.putText(frame, 'GT_int: '     + GT_int,                 (750 , 15), font, 1, (0  , 110, 255), 2, cv2.LINE_4)
        cv2.circle(frame, (int(xc), int(yc)), radius=5, color=(0, 255, 255), thickness=-1)
        frame = cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        #cv2.imwrite(''./image_debug/' + str(video_nu_all[0]).zfill(4) + '_' + str(temp_frame_all[y]).zfill(5) + '.png', frame)

def debug_input_img_sequence(cfg, img_path, img):
    """
    Function to visualise input image transformation. It is the actual image
    that is input into the network.
    """
    
    img = img.detach().cpu()
    
    # If normalized, unnormalize first (optional)
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std  = torch.tensor([0.229, 0.224, 0.225])
    # img = img * std[:, None, None] + mean[:, None, None]
    # img = img.clamp(0, 1)
    
    img = img.permute(1, 2, 0)  # [H, W, C]
    
    # Directory for saving images
    input_img_tansformed_path_dir = Path("./output/debug/input/input_img_tansformed")
    input_img_tansformed_path_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize image filename
    img_path = img_path.replace(cfg.data.dataset_folder_path, "")
    img_path = img_path.replace("/", "_")
    img_filename = Path(img_path)
    save_path = input_img_tansformed_path_dir / img_filename
    
    # Plot and save
    plt.figure(figsize=(6,6))
    plt.imshow(img)        # make sure img is [H, W, C] and in [0,1] or uint8
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    #print("Saved transformed image to:", save_path)
