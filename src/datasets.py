import torch
from torch.utils.data import Dataset
import numpy as np

class DJTransitionDataset(Dataset):
    def __init__(self, audio_data, cue_points, bpm_data, S_truth_data, target_bpm, sr, hop_length, segment_duration):
        """
        Args:
            audio_data (list of tuples): List of (S1_audio_signal, S2_audio_signal) tuples.
            cue_points (list of tuples): List of (cue_out_time_s1, cue_in_time_s2) tuples.
            bpm_data (list of tuples): List of (bpm_orig_s1, bpm_orig_s2) tuples.
            S_truth_data (list): List of mel-spectrograms for the real DJ transitions.
            target_bpm (float): Target BPM for time stretching.
            sr (int): Sampling rate.
            hop_length (int): Hop length for mel-spectrogram computation.
            segment_duration (float): Duration of the segment to extract in seconds.
        """
        self.audio_data = audio_data
        self.cue_points = cue_points
        self.bpm_data = bpm_data
        self.S_truth_data = S_truth_data
        self.target_bpm = target_bpm
        self.sr = sr
        self.hop_length = hop_length
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        S1_audio_signal, S2_audio_signal = self.audio_data[idx]
        cue_out_time_s1, cue_in_time_s2 = self.cue_points[idx]
        bpm_orig_s1, bpm_orig_s2 = self.bpm_data[idx]
        S_truth = self.S_truth_data[idx]

        input_tensor, S_truth_tensor = prepare_input_and_truth_label(S1_audio_signal, S2_audio_signal, S_truth,
                                                                     cue_out_time_s1, cue_in_time_s2,
                                                                     bpm_orig_s1, bpm_orig_s2, self.target_bpm,
                                                                     self.segment_duration, self.sr, self.hop_length)
        return input_tensor, S_truth_tensor

# Example usage
if __name__ == "__main__":
    # Dummy data for illustration purposes
    audio_data = [(np.zeros(int(44100*60)), np.zeros(int(44100*60)))]  # 60 seconds of silence for S1 and S2
    cue_points = [(30.0, 35.0)]  # Cue points in seconds
    bpm_data = [(100.0, 130.0)]  # Original BPMs of S1 and S2
    S_truth_data = [np.zeros((128, compute_num_frames(15, 44100, 512)))]  # Placeholder mel-spectrogram
    target_bpm = 120.0
    sr = 44100
    hop_length = 512
    segment_duration = 15.0

    dataset = DJTransitionDataset(audio_data, cue_points, bpm_data, S_truth_data, target_bpm, sr, hop_length, segment_duration)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Dummy model and optimizer
    model = TransitionPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    train_model(model, train_loader, optimizer, epochs=10, device='cpu')