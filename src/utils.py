import os

def create_dirs(experiment_name):
    experiments_folder = "../s194323/experiments"
    new_folder = os.path.join(experiments_folder, experiment_name)
    os.makedirs(new_folder, exist_ok=True)
    os.makedirs(os.path.join(new_folder, "weights"), exist_ok=True)
    os.makedirs(os.path.join(new_folder, "samples"), exist_ok=True)