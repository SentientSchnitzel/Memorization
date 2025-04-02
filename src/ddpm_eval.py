import os
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import json

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from data import CheXpertDataset


# custom imports
from ddpm import Diffusion
#from model import Classifier, UNet
from UNet import UNet


#from dataset.helpers import *
from utils import set_seed#, prepare_dataloaders
set_seed()

# Rembember to have the 
class VGG(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, features=False):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        x = self.dropout(self.flatten(feat))
        x = self.fc(x)
        if features:
            return feat
        else:
            return x


def convert_grayscale_to_rgb(images):
    # Repeat the single channel across the 3 channels: [batch_size, 1, height, width] -> [batch_size, 3, height, width]
    return images.repeat(1, 3, 1, 1)


def get_features(model, images, greyscale=False):
    model.eval()  
    with torch.no_grad():
        if greyscale:
            # NOTE: we are 
            images = convert_grayscale_to_rgb(images)
        features = model(images, features=True)
    features = features.squeeze(3).squeeze(2).cpu().numpy()
    return features

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    # https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
    # HINT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    # Implement FID score


    # 1) Compute the difference in means and its square norm
    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    # 2) Compute the sqrt of the product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical error can give slight imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 3) Compute trace component
    trace_component = np.trace(sigma1 + sigma2 - 2 * covmean)

    fid = diff_squared + trace_component

    return fid



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    #parser.add_argument('--model_type', type=str, default='default.yaml', help='config file')
    parser.add_argument('--experiment_name', type=str, default=None, help='Fine-tune path')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    return parser.parse_args()



def main(num_classes=14):
    # Aron: source /work3/s194262/adv_dl_cv/bin/activate

    # python ddpm_eval.py --experiment_name=test_model --fast_run=False
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Set up
    # Parse arguments
    args = parse_args()

    cfg = True if 'cfg' in args.experiment_name else False
    shared_folder = "/work3/s194262/GitHub/Memorization/s194323/"

    # Load the config file
    with open(os.path.join(shared_folder, f"experiments/{args.experiment_name}/weights/config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    T=config['T']
    img_size=config['img_size']
    input_channels=config['input_channels']
    channels=config['channels']
    time_dim=config['time_dim']
    batch_size=config['batch_size']
    lr=config['lr']
    num_epochs=config['num_epochs']
    experiment_name="ddpm"
    train_frac=config['train_frac']
    cfg=config['cfg']
    num_classes=config['num_classes']



    #############################################################################
    # initialize model
    #############################################################################
    
    # Load the UNet model
    num_classes = num_classes if cfg else None

    unet_ddpm = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 num_classes=num_classes, time_dim=time_dim,channels=channels, device=device
    )
    unet_ddpm.to(device)
    # load model weights
    model_path = os.path.join(shared_folder, f"experiments/{args.experiment_name}/weights/model.pth")
    unet_ddpm.load_state_dict(torch.load(
        model_path,
        map_location=device)
    )
    # # Loading the whole model
    # unet_ddpm = torch.load(
    #     os.path.join(shared_folder, f"experiments/{args.experiment_name}/weights/whole_model.pth"),
    #     map_location=device
    # )
    unet_ddpm.eval()


    # Initialize the diffusion model
    diff_type = 'DDPM-cFg' if cfg else 'DDPM'
    ddpm = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, diff_type=diff_type, device=device)

    #############################################################################
    # VGG
    #############################################################################
    
    vgg = VGG()
    vgg.to(device)
    vgg.eval()
    vgg_model_path = os.path.join(shared_folder, 'models/VGG/model.pth')
    vgg.load_state_dict(torch.load(
        vgg_model_path,
        map_location=device)
    )
    dims = 256 # VGG feature dim
    # TODO: ok?
    vgg_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])


    #############################################################################
    # Load dataset
    #############################################################################

    # TODO: handle fast run here

    # TODO: need a test set?

    data_root = os.path.join(shared_folder, 'data')
    dataset = CheXpertDataset(
        csv_file = f"{data_root}/train.csv",
        img_root_dir = f"{data_root}/train",
        frac=train_frac,)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #############################################################################
    # Compute FID score
    #############################################################################

    # Initialize arrays to store features
    original_features = np.empty((len(data_loader.dataset), dims))
    generated_features = np.empty((len(data_loader.dataset), dims))
    all_labels = np.empty((len(data_loader.dataset), config['num_classes']))

    start_idx = 0
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Original images
        original_feat = get_features(vgg, images, greyscale=True)
        original_features[start_idx:start_idx + original_feat.shape[0]] = original_feat

        # And labels
        all_labels[start_idx:start_idx + labels.shape[0]] = labels.cpu().numpy()

        # Generated images
        if not cfg:
            # not classifier-free guidance
            generated_images = ddpm.p_sample_loop(unet_ddpm, batch_size=images.shape[0], verbose=False)
        else:
            # TODO: handle appropriately

            # classifier-free guidance
            y = F.one_hot(y, num_classes=5).float()
            generated_images = ddpm.p_sample_loop(unet_ddpm, batch_size=images.shape[0], y=y, verbose=False)
        
        generated_images = vgg_transform(generated_images/255.0)
        # NOTE: handling greyscale images by repeating the channel
        generated_feat = get_features(vgg, generated_images, greyscale=True)

        # store features
        generated_features[start_idx:start_idx + generated_feat.shape[0]] = generated_feat

        start_idx = start_idx + original_feat.shape[0]
    
    # Identify NaN or Inf values
    nan_mask = np.isnan(original_features) | np.isnan(generated_features) #| np.isnan(all_labels)
    inf_mask = np.isinf(original_features) | np.isinf(generated_features) #| np.isinf(all_labels)

    # Combine masks
    invalid_mask = nan_mask.any(axis=1) | inf_mask.any(axis=1)

    print(f"Number of invalid observation: {np.sum(invalid_mask)}")

    # Remove invalid rows
    original_features = original_features[~invalid_mask]
    generated_features = generated_features[~invalid_mask]
    all_labels = all_labels[~invalid_mask]


    # FID results
    fid_results = {}
    mu_original, sigma_original = feature_statistics(original_features)
    mu_generated, sigma_generated = feature_statistics(generated_features)


    # Overall FID score
    fid_score = frechet_distance(mu_original, sigma_original, mu_generated, sigma_generated)
    print(f'[FID score] {fid_score:.3f}\n')
    fid_results['fid_score'] = fid_score


    # FID per class
    fid_scores_per_class = {}
    for class_id in range(config['num_classes']):
        # Extract features for the current class
        class_indices = np.where(all_labels[:, class_id] == 1)[0]
        
        original_features_class = original_features[class_indices]
        generated_features_class = generated_features[class_indices]

        # Handle insufficient samples
        if original_features_class.shape[0] < 2 or generated_features_class.shape[0] < 2:
            print(f"    Skipping class {class_id} due to insufficient samples")
            continue

        # Compute statistics for original and generated features for the class
        mu_original_class, sigma_original_class = feature_statistics(original_features_class)
        mu_generated_class, sigma_generated_class = feature_statistics(generated_features_class)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(sigma_original_class)) or np.any(np.isnan(sigma_generated_class)):
            print(f"    NaN detected in covariance matrices for class {class_id}")
            continue
        if np.any(np.isinf(sigma_original_class)) or np.any(np.isinf(sigma_generated_class)):
            print(f"    Inf detected in covariance matrices for class {class_id}")
            continue


        # Compute FID for the class
        fid_score_class = frechet_distance(mu_original_class, sigma_original_class, mu_generated_class, sigma_generated_class)
        fid_scores_per_class[class_id] = fid_score_class


        # TODO: translate class number to class names
        print(f'    [FID score for class {class_id}] {fid_score_class:.3f}')
        fid_results[f'fid_score_class_{class_id}'] = fid_score_class
    

    # The average FID score class-wise
    average_fid_score = np.mean(list(fid_scores_per_class.values()))
    print(f'[Average FID score across all classes] {average_fid_score:.3f}')
    fid_results['average_fid_score'] = average_fid_score


    # Save the results
    # create a evaluation folder
    eval_folder = os.path.join(shared_folder, f"experiments/{args.experiment_name}/evaluation")
    os.makedirs(eval_folder, exist_ok=True)
    # save the fid score as json
    fid_results_path = os.path.join(shared_folder, f"experiments/{args.experiment_name}/evaluation/fid_results.json")
    with open(fid_results_path, 'w') as f:
        json.dump(fid_results, f)



if __name__ == '__main__':
    main()