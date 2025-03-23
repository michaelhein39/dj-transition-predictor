import os
import torch
from torch.utils.data import Dataset


class DJTransitionDataset(Dataset):
    def __init__(self, preprocessed_dir):
        """
        Args:
            preprocessed_dir (str): Directory with all the preprocessed .pt files.
        """
        self.preprocessed_dir = preprocessed_dir

        # Filter and store the full paths of .pt files only
        self.file_list = [os.path.join(preprocessed_dir, fname)
                          for fname in os.listdir(preprocessed_dir) if fname.endswith('.pt')]

        # Validate that the directory is not empty
        if not self.file_list:
            raise ValueError(f"No preprocessed files found in directory: {preprocessed_dir}")

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (input_tensor, S_truth_tensor) loaded from the preprocessed file.
        """    
        file_path = self.file_list[idx]
        try:
            # Load the preprocessed tensors from the .pt file
            input_tensor, S_truth_tensor = torch.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

        return input_tensor, S_truth_tensor


# Use this for single sample training to demonstrate the model
class SingleSampleDataset(Dataset):
    def __init__(self, preprocessed_dir, n_repeats=1000):
        """
        Args:
            preprocessed_dir (str): Directory with the single preprocessed .pt file.
            n_repeats (int): Number of times to repeat the sample.
        """
        self.file_list = [os.path.join(self.preprocessed_dir, fname)
                          for fname in os.listdir(self.preprocessed_dir) if fname.endswith('.pt')]

        if not self.file_list:
            raise ValueError(f"No preprocessed files found in directory: {self.preprocessed_dir}")

        if len(self.file_list) > 1:
            raise ValueError(f"Multiple files found in directory: {self.preprocessed_dir}")

        self.n_repeats = n_repeats

    def __len__(self):
        return self.n_repeats

    def __getitem__(self, idx):
        file_path = self.file_list[0]
        try:
            input_tensor, S_truth_tensor = torch.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")
        return input_tensor, S_truth_tensor