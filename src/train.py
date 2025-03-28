import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from torch import optim
import logging

from UNet import UNet
from ddpm import Diffusion
from data import CheXpertDataset
from utils import create_dirs

def train(T=1000, img_size=224, input_channels=1, channels=16, 
          time_dim=256, batch_size=16, lr=1e-3, num_epochs=5, device='cpu',
          experiment_name="ddpm", train_frac=None, cfg=False, num_classes=None):

    create_dirs(experiment_name)
    
    num_classes = num_classes if cfg else None

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 num_classes=num_classes, time_dim=time_dim,channels=channels, device=device).to(device)
    
    diff_type = 'DDPM-cFg' if cfg else 'DDPM'
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, diff_type=diff_type, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    
    data_root = "../s194323/data/"
    train_dataset = CheXpertDataset(
        csv_file = f"{data_root}/train.csv",
        img_root_dir = f"{data_root}/train",
        frac=train_frac)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    l = len(train_loader)

    min_train_loss = 1e10
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        epoch_loss = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            
            if diff_type == 'DDPM-cFg':
                # one-hot encode labels for classifier-free guidance
                labels = labels.to(device)
                labels = F.one_hot(labels, num_classes=num_classes).float()
            else :
                labels = None
            
            p_uncod = 0.1
            if np.random.rand() < p_uncod:
                labels = None
            
            t = torch.randint(0, T, (images.size(0),), device=device).long()
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t, y = labels)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix(MSE=loss.item())

        epoch_loss /= l
        if epoch_loss <= min_train_loss:
            torch.save(model.state_dict(), os.path.join("../s194323/experiments/", experiment_name, "weights" ,f"model.pth"))
            min_train_loss = epoch_loss    
        
        if diffusion.diff_type == 'DDPM-cFg':
            y = torch.tensor([np.random.randint(0,5)], device=device)
            title = f'Epoch {epoch} with label:{CLASS_LABELS[y.item()]}'
            y = F.one_hot(y, num_classes=num_classes).float()
        else:
            y = None
            title = f'Epoch {epoch}'
        
        sampled_images = diffusion.p_sample_loop(model, batch_size=images.shape[0], y=y)
        
        # save the first sampled image
        img_name = os.path.join("../s194323/experiments/", experiment_name, "samples", f"sampled_image_epoch_{epoch}.png")
        plt.imsave(img_name, sampled_images[0].squeeze().cpu().numpy().transpose(1, 2, 0))           

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    num_epochs = 5
    train_frac = 0.25

    train(T = 1000, device=device, num_epochs=num_epochs, train_frac=train_frac, experiment_name="ddpm_2")