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

        input_tensor, S_truth_tensor = create_input_and_label(S1_audio_signal, S2_audio_signal, S_truth,
                                                              cue_out_time_s1, cue_in_time_s2,
                                                              bpm_orig_s1, bpm_orig_s2, self.target_bpm,
                                                              self.segment_duration, self.sr, self.hop_length)
        return input_tensor, S_truth_tensor


############################################################
# Helper Functions
############################################################

def create_input_and_label(S1_audio_signal, S2_audio_signal, S_truth, 
                                  cue_out_time_s1, cue_in_time_s2,
                                  bpm_orig_s1, bpm_orig_s2,
                                  bpm_target, segment_duration=SEGMENT_DURATION,
                                  sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Prepare a single input batch and corresponding truth label for training.

    Time stretch the two input audio signals to the target BPM, convert to mel-spectrograms,
    extract a 15-second segment around the cue points, and pad if necessary.

    Finally, convert all to torch tensors and combine the two segments as channels.

    Args:
        S1_audio_signal: raw audio array for track 1
        S2_audio_signal: raw audio array for track 2
        S_truth: mel-spectrogram for the real DJ transition
        cue_out_time_s1: the cue-out time (in seconds) for S1
        cue_in_time_s2: the cue-in time (in seconds) for S2
        bpm_orig_s1: original BPM of S1
        bpm_orig_s2: original BPM of S2
        bpm_target: target BPM for time stretching
        segment_duration: duration of the segment to extract in seconds
        sr: sampling rate
        hop_length: hop length for mel-spectrogram computation

    Returns:
        input_tensor: torch tensor of shape (batch=1, 2, N_MELS, T)
        S_truth_tensor: torch tensor of shape (batch=1, N_MELS, T)
    """

    # Time stretch S1 and S2 to target BPM
    S1_stretched = adjust_bpm(S1_audio_signal, bpm_orig_s1, bpm_target)
    S2_stretched = adjust_bpm(S2_audio_signal, bpm_orig_s2, bpm_target)

    # Convert to mel-spectrograms
    S1_mel = ft.audio_signal_melspectrogram(S1_stretched, sr)
    S2_mel = ft.audio_signal_melspectrogram(S2_stretched, sr)

    # Compute frame indices of cue points after time stretching
    cue_out_frame_s1 = convert_beat_time_to_frame(cue_out_time_s1, bpm_orig_s1, bpm_target, sr, hop_length)
    cue_in_frame_s2 = convert_beat_time_to_frame(cue_in_time_s2, bpm_orig_s2, bpm_target, sr, hop_length)
    
    # Determine how many frames correspond to segment_duration seconds
    total_frames = compute_num_frames(segment_duration, sr, hop_length) 
    
    # Determine necessary padding
    z_frames = S_truth.shape[1]  # original number of ground truth transition frames
    if z_frames < total_frames:
        # x and y split the remaining time
        remainder = total_frames - z_frames
        x_frames = remainder // 2
        y_frames = remainder - x_frames
    else:
        # If z_frames == total_frames just set x_frames = y_frames = 0
        x_frames = 0
        y_frames = 0

    # Extract segments from S1 and S2
    # S1 segment: start from cue_out_frame_s1 - x_frames and go for total_frames
    start_s1 = cue_out_frame_s1 - x_frames
    end_s1 = start_s1 + total_frames

    start_s2 = cue_in_frame_s2 - (total_frames - y_frames)
    end_s2 = start_s2 + total_frames

    S1_segment = segment_melspectrogram(S1_mel, start_s1, end_s1, total_frames)
    S2_segment = segment_melspectrogram(S2_mel, start_s2, end_s2, total_frames)

    # Convert all to torch tensors
    S1_tensor = torch.tensor(S1_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, T)
    S2_tensor = torch.tensor(S2_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, T)
    S_truth_tensor = torch.tensor(S_truth, dtype=torch.float32).unsqueeze(0) # add batch dim

    # Combine S1 and S2 as channels: (batch=1, 2, N_MELS, T)
    input_tensor = torch.cat([S1_tensor, S2_tensor], dim=0).unsqueeze(0)

    return input_tensor, S_truth_tensor


def adjust_bpm(audio_signal, bpm_orig, bpm_target):
    """
    Time-stretching the audio signal so that its tempo matches bpm_target.
    """
    # Calculate scaling factor to adjust the BPM
    time_stretch_factor = bpm_orig / bpm_target

    # Use librosa's time_stretch (phase vocoder) to adjust BPM of audio_signal
    # without altering pitch
    stretched_audio_signal = librosa.effects.time_stretch(audio_signal, rate=1.0/time_stretch_factor)
    return stretched_audio_signal


def convert_beat_time_to_frame(beat_time, bpm_orig, bpm_target, sr, hop_length):
    """
    Convert an original beat time to the corresponding frame index in the
    new time-stretched mel-spectrogram.

    Parameters:
        beat_time (float): Original beat time (in seconds).
        bpm_orig (float): Original tempo (in BPM).
        bpm_target (float): Target tempo of stretched audio (in BPM).
        sr (int): Sampling rate (e.g., 22050 Hz).
        hop_length (int): Hop length for mel-spectrogram (e.g., 512 samples).

    Returns:
        int: Frame index in the resampled mel-spectrogram.
    """
    # Adjust beat time for the resampled tempo
    scaling_factor = bpm_orig / bpm_target
    scaled_time = beat_time * scaling_factor

    # Convert scaled time to frame index in the time-stretched mel-spectrogram
    frame_index = (scaled_time * sr) // hop_length

    return frame_index


def compute_num_frames(duration_sec, sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Given a duration in seconds, compute how many STFT frames correspond to that duration.
    Number of frames = floor(duration_sec * sr / hop_length)

    This equates to 645 frames for 15 seconds.
    """
    num_frames = (duration_sec * sr) // hop_length
    return num_frames


def segment_melspectrogram(S, start, end, total_frames):
    """
    Extract a segment of a mel-spectrogram, padding as necessary to ensure a segment
    of exactly total_frames frames.

    Parameters:
        S (numpy.ndarray): Mel-spectrogram with shape (n_mels, n_frames).
        start (int): Start frame index (inclusive) of the segment.
        end (int): End frame index (exclusive) of the segment.
        total_frames (int): Total number of frames in the output segment.

    Returns:
        numpy.ndarray: Segment of the mel-spectrogram with shape (n_mels, total_frames).
    """
    n_mels, n_frames = S.shape

    if start < 0 and end > n_frames:
        raise ValueError("Start frame is negative and end frame is greater than total frames.")

    # This assumes that only one of these conditions is true, when in reality
    # it's possible for both to be true.
    if start < 0:
        # Pad on the left
        left_pad = abs(start)
        S_segment = np.hstack([np.zeros((n_mels, left_pad)), S[:, :end]])
    elif end > n_frames:
        # Pad on the right
        right_pad = end - n_frames
        S_segment = np.hstack([S[:, start:], np.zeros((n_mels, right_pad))])
    else:
        # No padding
        S_segment = S[:, start:end]

    # Ensure the segment has exactly total_frames frames
    if S_segment.shape[1] < total_frames:
        raise ValueError("Segment does not have enough frames.")
    
    return S_segment