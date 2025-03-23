import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TransitionPredictor
from train import train_model
from datasets import DJTransitionDataset

def main():
    # Initialize model
    model = TransitionPredictor()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

    # Dataset and DataLoader
    preprocessed_dir = 'data/preprocessed'  # Adjust the path as necessary
    dataset = DJTransitionDataset(preprocessed_dir)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)  # Adjust batch_size as needed

    # Train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, optimizer, epochs=10, device=device)

if __name__ == '__main__':
    main()