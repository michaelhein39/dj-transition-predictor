import os
import torch
from torch.utils.data import Dataset


SAMPLING_RATE = 22050       # Sampling rate for audio
HOP_LENGTH = 512            # Hop length for STFT -> mel-spectrogram
SEGMENT_DURATION = 15.0     # 15 seconds segment length


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