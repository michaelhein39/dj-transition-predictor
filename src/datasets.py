"""
Pseudocode for data loading.

Assume you have:
    - S1, S2: raw audio arrays for track 1 and 2
    - S_truth: mel-spectrogram for the real DJ transition
    - cue_out_time_s1: the cue-out time (in seconds) for S1
    - cue_in_time_s2: the cue-in time (in seconds) for S2
    - bpm_orig_s1, bpm_orig_s2: original BPMs of S1 and S2
Steps:
    1) Resample S1 and S2 to TARGET_BPM
    2) Extract mel-spectrograms
    3) Compute frame indices for cue points using beat_time_to_resampled_frame
    4) Extract 15-second segment around the transition
    5) Prepare input batch and S_truth

"""

import torch
import librosa
import numpy as np
import lib.feature as ft
from torch.utils.data import Dataset


SAMPLING_RATE = 22050       # Sampling rate for audio
HOP_LENGTH = 512            # Hop length for STFT -> mel-spectrogram
SEGMENT_DURATION = 15.0     # 15 seconds segment length


class DJTransitionDataset(Dataset):
    def __init__(self, audio_data, cue_points, bpm_data, S_truth_data, sr, hop_length, segment_duration):
        """
        Args:
            audio_data (list of tuples): List of (S1_audio_signal, S2_audio_signal) tuples.
            cue_points (list of tuples): List of (cue_out_time_s1, cue_in_time_s2) tuples.
            bpm_data (list of tuples): List of (bpm_orig_s1, bpm_orig_s2, target_bpm) tuples.
            S_truth_data (list): List of mel-spectrograms for the real DJ transitions.
            sr (int): Sampling rate.
            hop_length (int): Hop length for mel-spectrogram computation.
            segment_duration (float): Duration of the segment to extract in seconds.
        """
        self.audio_data = audio_data
        self.cue_points = cue_points
        self.bpm_data = bpm_data
        self.S_truth_data = S_truth_data
        self.sr = sr
        self.hop_length = hop_length
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        S1_audio_signal, S2_audio_signal = self.audio_data[idx]
        cue_out_time_s1, cue_in_time_s2 = self.cue_points[idx]
        bpm_orig_s1, bpm_orig_s2, target_bpm = self.bpm_data[idx]
        S_truth = self.S_truth_data[idx]

        input_tensor, S_truth_tensor = create_input_and_label(S1_audio_signal, S2_audio_signal, S_truth,
                                                              cue_out_time_s1, cue_in_time_s2,
                                                              bpm_orig_s1, bpm_orig_s2, target_bpm,
                                                              self.segment_duration, self.sr, self.hop_length)
        return input_tensor, S_truth_tensor