from utils.check_point import load_config
from train.trainer import train_model



if __name__ == "__main__":
    config = load_config("config/config.json")
    
    if config["mode"] == "train":
        train_model(config)
    elif config["mode"] == "validate":
        validate_model(config)
    elif config["mode"] == "infer":
        run_inference(config)
