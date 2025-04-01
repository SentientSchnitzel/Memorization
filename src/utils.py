import os
import yaml
    
def load_config(config_name):
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