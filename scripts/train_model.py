import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models import TransitionPredictor
from src.train import train_model
from src.datasets import DJTransitionDataset, SingleSampleDataset

def main():
    # Initialize model
    model = TransitionPredictor()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Adjust learning rate as needed

    # Dataset and DataLoader, using a single sample to demonstrate model
    preprocessed_dir = 'data/preprocessed'
    # dataset = DJTransitionDataset(preprocessed_dir)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)  # Adjust batch_size as needed
    dataset = SingleSampleDataset(preprocessed_dir, n_repeats=1000)  # Repeat the single sample 1000 times
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)  # Use batch size of 1

    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, optimizer, epochs=10, device=device, save_dir='models')

if __name__ == '__main__':
    main()