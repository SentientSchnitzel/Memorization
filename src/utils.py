import os
import yaml
    
def load_config(config_name):
    # if empty
    if not config_name:
        return {}

    config_path = os.path.join("../configs", config_name + ".yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def save_config(config, save_dir):
    save_path = os.path.join(save_dir, "config.yaml")
    with open(save_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

def create_dirs(experiment_name):
    experiments_folder = "../s194323/experiments"
    new_folder = os.path.join(experiments_folder, experiment_name)
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, "weights"), exist_ok=True)
    os.makedirs(os.path.join(new_folder, "samples"), exist_ok=True)
    
CLASS_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"]






import os

def create_dirs(experiment_name):
    experiments_folder = "../s194323/experiments"
    new_folder = os.path.join(experiments_folder, experiment_name)
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, "weights"), exist_ok=True)
    os.makedirs(os.path.join(new_folder, "samples"), exist_ok=True)
    
CLASS_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
            "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"]



import torch
import random
import numpy as np


SEED = 42
def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

