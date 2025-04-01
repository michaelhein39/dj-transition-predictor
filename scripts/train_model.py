import os
import random
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from src.models import TransitionPredictor
from src.train import train_model
from src.datasets import DJTransitionDataset, SingleSampleDataset


def run_training(lr, epochs, model_save_name, seed):
    # Set the seed for reproducibility
    set_seed(seed)

    # Initialize model
    model = TransitionPredictor()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate as needed

    # Dataset and DataLoader, using a single sample to demonstrate model
    preprocessed_dir = 'data/preprocessed'
    # dataset = DJTransitionDataset(preprocessed_dir)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)  # Adjust batch_size as needed
    dataset = SingleSampleDataset(preprocessed_dir, n_repeats=1000)  # Repeat the single sample 1000 times
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)  # Use batch size of 1

    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_array = train_model(model, train_loader, optimizer, model_save_name, seed,
                             epochs=epochs, device=device, save_dir='models')
    
    return loss_array


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    lr = 1e-5
    epochs = 10
    model_save_name = 'mel_lr1e-5'
    seed = 2025
    loss_array = run_training(lr, epochs, model_save_name, seed)

    # Save the loss array to a file
    os.makedirs('losses', exist_ok=True)
    path = os.path.join('losses', f'{model_save_name}_loss_array.npy')
    np.save(path, loss_array)
    print(f"Loss array saved at {path}")