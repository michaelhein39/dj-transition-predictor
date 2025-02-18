import os
import torch
from torch.utils.data import Dataset


class DJTransitionDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        self.file_list = os.listdir(preprocessed_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.preprocessed_dir, self.file_list[idx]))
        input_tensor, S_truth_tensor = data
        return input_tensor, S_truth_tensor