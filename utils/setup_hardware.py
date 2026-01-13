import torch
import os



def get_devie(cfg, cmd_args):
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i))
    
    if cfg.system.multi_gpu == True:
        cfg.system.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("torch.cuda.get_device_properties(cfg.system.device)", torch.cuda.get_device_properties(cfg.system.device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= '0, 1'
        if cmd_args.device == 0:
            cfg.system.device = torch.device('cuda:0')
        elif cmd_args.device == 1:
            cfg.system.device = torch.device('cuda:1')
        print('device:', cfg.system.device)
        print("torch.cuda.get_device_properties(cfg.system.device)", torch.cuda.get_device_properties(cfg.system.device))