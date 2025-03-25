"""
Pseudocode for data preprocessing.

Assume you have:
    - S1, S2: raw audio arrays for track 1 and 2
    - S_truth: mel-spectrogram for the real DJ transition
    - cue_out_time_s1: the cue-out time (in seconds) for S1
    - cue_in_time_s2: the cue-in time (in seconds) for S2
    - bpm_orig_s1, bpm_orig_s2: original BPMs of S1 and S2
Steps:
    1) Time stretch S1 and S2 to target BPM
    2) Extract mel-spectrograms
    3) Compute frame indices from cue points
    4) Extract (SEGMENT_DURATION)-second segment around the transition
    5) Save input and label pairs

"""


import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import lib.feature as ft
from lib.constants import *


def main(csv_file, save_dir, overwrite=False):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing", unit="sample"):
        S1_audio_path = row['path_S1']
        S2_audio_path = row['path_S2']
        mix_audio_path = row['mix_path']

        cue_out_time_S1 = row['cue_out_time_S1']
        cue_in_time_S2 = row['cue_in_time_S2']
        cue_out_time_mix = row['cue_out_time_mix']
        cue_in_time_mix = row['cue_in_time_mix']

        bpm_orig_S1 = row['bpm_orig_S1']
        bpm_orig_S2 = row['bpm_orig_S2']
        bpm_target = row['bpm_target']

        # Skip if file already exists and overwrite is not enabled
        file_name = f"sample_{idx}.pt"
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path) and not overwrite:
            print(f"Skipping {file_name}, already exists.")
            continue

        S1_audio_signal, _ = librosa.load(S1_audio_path, sr=SAMPLING_RATE)
        S2_audio_signal, _ = librosa.load(S2_audio_path, sr=SAMPLING_RATE)
        mix_audio_signal, _ = librosa.load(mix_audio_path, sr=SAMPLING_RATE)

        # Create input and truth tensors
        input_tensor, S_truth_tensor = generate_training_tensors(
            S1_audio_signal, S2_audio_signal, mix_audio_signal,
            cue_out_time_S1, cue_in_time_S2,
            cue_out_time_mix, cue_in_time_mix,
            bpm_orig_S1, bpm_orig_S2, bpm_target
        )
        
        # Save the processed data
        torch.save((input_tensor, S_truth_tensor), file_path)


def generate_training_tensors(S1_audio_signal, S2_audio_signal, mix_audio_signal, 
                              cue_out_time_s1, cue_in_time_s2,
                              cue_out_time_mix, cue_in_time_mix,
                              bpm_orig_s1, bpm_orig_s2,
                              bpm_target, segment_duration=SEGMENT_DURATION,
                              sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Prepare a single input batch and corresponding truth label for training.

    Time stretch the two input audio signals to the target BPM, convert to mel-spectrograms,
    extract a (SEGMENT_DURATION)-second segment around the cue points, and pad if necessary.

    Finally, convert all to torch tensors and combine the two segments as channels.

    Args:
        S1_audio_signal: raw audio array for track 1
        S2_audio_signal: raw audio array for track 2
        mix_audio_signal: raw audio array for the mix containing the real DJ transition
        cue_out_time_s1: the cue-out time (in seconds) for S1
        cue_in_time_s2: the cue-in time (in seconds) for S2
        cue_out_time_mix: the cue-out time (in seconds) for the mix (beginning of transition)
        cue_in_time_mix: the cue-in time (in seconds) for the mix (end of transition)
        bpm_orig_s1: original BPM of S1
        bpm_orig_s2: original BPM of S2
        bpm_target: target BPM (from the mix) for time stretching
        segment_duration: duration of the segment to extract in seconds
        sr: sampling rate
        hop_length: hop length for mel-spectrogram computation

    Returns:
        input_tensor: torch tensor of shape (batch=1, 2, N_MELS, F)
        S_truth_tensor: torch tensor of shape (batch=1, N_MELS, F)
    """

    # Time stretch S1 and S2 to target BPM
    S1_stretched = adjust_bpm(S1_audio_signal, bpm_orig_s1, bpm_target)
    S2_stretched = adjust_bpm(S2_audio_signal, bpm_orig_s2, bpm_target)

    # Convert to mel-spectrograms
    S1_mel = ft.audio_signal_melspectrogram(S1_stretched, sr)
    S2_mel = ft.audio_signal_melspectrogram(S2_stretched, sr)
    mix_mel = ft.audio_signal_melspectrogram(mix_audio_signal, sr)

    # Compute frame indices of cue points after time stretching
    cue_out_frame_s1 = convert_beat_time_to_frame(cue_out_time_s1, bpm_orig_s1, bpm_target, sr, hop_length)
    cue_in_frame_s2 = convert_beat_time_to_frame(cue_in_time_s2, bpm_orig_s2, bpm_target, sr, hop_length)
    cue_out_frame_mix = convert_beat_time_to_frame(cue_out_time_mix, bpm_target, bpm_target, sr, hop_length)
    cue_in_frame_mix = convert_beat_time_to_frame(cue_in_time_mix, bpm_target, bpm_target, sr, hop_length)

    # Original number of ground truth transition frames
    z_frames = cue_in_frame_mix - cue_out_frame_mix + 1

    # Number of frames corresponding to segment_duration seconds
    total_frames = compute_num_frames(segment_duration, sr, hop_length)

    # Determine necessary padding
    if z_frames < total_frames:
        # x and y split the remaining time
        remainder = total_frames - z_frames
        x_frames = int(remainder / 2)
        y_frames = remainder - x_frames
    else:
        # If z_frames == total_frames just set x_frames = y_frames = 0
        x_frames = 0
        y_frames = 0

    # Extract segments from S1, S2, and mix
    # S1 segment: start from cue_out_frame_s1 - x_frames and go for total_frames
    start_s1 = cue_out_frame_s1 - x_frames
    end_s1 = start_s1 + total_frames

    start_s2 = cue_in_frame_s2 - (total_frames - y_frames)
    end_s2 = start_s2 + total_frames

    start_mix = cue_out_frame_mix - x_frames
    end_mix = cue_in_frame_mix + y_frames + 1
    if end_mix - start_mix != total_frames:
        raise ValueError("Padding issue")

    S1_segment = segment_melspectrogram(S1_mel, start_s1, end_s1, total_frames)
    S2_segment = segment_melspectrogram(S2_mel, start_s2, end_s2, total_frames)
    S_truth_segment = segment_melspectrogram(mix_mel, start_mix, end_mix, total_frames)

    # Convert all to torch tensors
    S1_tensor = torch.tensor(S1_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, F)
    S2_tensor = torch.tensor(S2_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, F)
    S_truth_tensor = torch.tensor(S_truth_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, F)

    # Combine S1 and S2 as channels: (batch=1, 2, N_MELS, F)
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
    frame_index = int(scaled_time * sr / hop_length)

    return frame_index


def compute_num_frames(duration_sec, sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Given a duration in seconds, compute how many STFT frames correspond to that duration.
    Number of frames = floor(duration_sec * sr / hop_length)

    This equates to 645 frames for a 15-second segment.
    """
    num_frames = int(duration_sec * sr / hop_length)
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
        raise ValueError("Desired padded segment is longer than the track itself.")

    # This assumes that only one of these conditions is true, when in reality
    # it's possible for both to be true.
    if start < 0:
        # Pad on the left due to negative start
        left_pad = abs(start)
        S_segment = np.hstack([np.zeros((n_mels, left_pad)), S[:, :end]])
    elif end > n_frames:
        # Pad on the right due to end > n_frames
        right_pad = end - n_frames
        S_segment = np.hstack([S[:, start:], np.zeros((n_mels, right_pad))])
    else:
        # No padding because start and end are valid in audio
        S_segment = S[:, start:end]

    # Ensure the segment has exactly total_frames frames
    if S_segment.shape[1] != total_frames:
        raise ValueError("Segment has wrong number of frames.")
    
    return S_segment


if __name__ == "__main__":
    # Path to the filtered CSV file
    # csv_file = 'data/filtered_input_output_pairs.csv'
    csv_file = 'data/segment/filtered_input_output_pairs.csv'

    # Directory to save the preprocessed data
    save_dir = 'data/preprocessed'

    main(csv_file, save_dir, overwrite=False)