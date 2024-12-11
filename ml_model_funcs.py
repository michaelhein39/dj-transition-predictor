import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import lib.feature as ft

############################################################
# Constants and Hyperparameters
############################################################
SAMPLING_RATE = 22050       # Sampling rate for audio
HOP_LENGTH = 512            # Hop length for STFT -> mel-spectrogram
N_MELS = 128                # Number of mel-frequency bins
SEGMENT_DURATION = 15.0     # 15 seconds segment length
VOLUME_CONTROL_SIGNAL_COUNT = 4     # 2 control signals (start, slope) * 2 tracks for volume
BANDPASS_CONTROL_SIGNAL_COUNT = 12  # 2 control signals (start, slope) * 3 bands * 2 tracks

# You may adjust these as needed based on your network design:
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
EPOCHS = 10

############################################################
# Utility Functions
############################################################

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

def pad_melspectrogram(S, target_frames, left_side=True):
    """
    Pad or truncate a mel-spectrogram S (shape: n_mels x frames) to have exactly target_frames in time dimension.
    If left_side is True, pad with zeros on the left. Otherwise pad on the right.
    """
    n_mels, curr_frames = S.shape
    if curr_frames == target_frames:
        # No padding or truncation needed
        return S
    if curr_frames > target_frames:
        # Needs to be truncated, not padded
        print(f'=> ERROR: mel-spectrogram is longer than target duration.')
        return None

    if left_side:
        # Pad on the left side with zeros
        pad_amount = target_frames - curr_frames
        S_padded = np.hstack([np.zeros((n_mels, pad_amount)), S])
    else:
        # Pad on the right side with zeros
        pad_amount = target_frames - curr_frames
        S_padded = np.hstack([S, np.zeros((n_mels, pad_amount))])
        
    return S_padded

def compute_num_frames(duration_sec, sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Given a duration in seconds, compute how many STFT frames correspond to that duration.
    Number of frames = floor(duration_sec * sr / hop_length)
    """
    num_frames = (duration_sec * sr) // hop_length
    return num_frames

############################################################
# Model Definition
############################################################

class TransitionPredictor(nn.Module):
    """
    CNN-based model to predict masking parameters from two input mel-spectrograms.
    The input will be a tensor of shape: (batch, 2, N_MELS, T)
    - 2 channels: one for S1, one for S2
    - N_MELS mel bins
    - T frames in the mel-spectrograms (corresponding to 15 seconds)
    """
    def __init__(self, n_mels=N_MELS, time_frames=645,  # time_frames should be computed dynamically
                 volume_control_signal_count=VOLUME_CONTROL_SIGNAL_COUNT, 
                 bandpass_control_signal_count=BANDPASS_CONTROL_SIGNAL_COUNT):
        super(TransitionPredictor, self).__init__()
        
        # Simple CNN (adjust as needed)
        # We will reduce temporal dimension and extract features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),  # reduce both freq and time dims

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # reduce to a single feature vector
        )

        # Fully connected layers to map features to parameters
        # We have total of (volume_control_signal_count + bandpass_control_signal_count) outputs
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, volume_control_signal_count + bandpass_control_signal_count)
        )

    def forward(self, x):
        # x shape: (batch, 2, N_MELS, T)
        features = self.conv_layers(x) # shape: (batch, 64, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 64)
        control_signals = self.fc(features)  # (batch, volume_control_signal_count + bandpass_control_signal_count)
        return control_signals
    
############################################################
# Masking Functions
############################################################

def prelu1(t, st, delta):
    """
    PReLU1 function (t - st)*delta clipped between 0 and 1
    
    Parameters:
    t (Tensor): time tensor
    st (float): start time
    delta (float): slope
    
    Returns:
    val (Tensor): clipped value
    """

    val = (t - st) * delta
    val = torch.clamp(val, min=0.0, max=1.0)
    return val

def apply_masks(S1, S2, control_signals, n_mels=128):
    """
    Given the predicted control signals and the original S1 and S2 spectrograms,
    apply volume and band masks using the defined PReLU-based fading functions.
    
    control_signals layout:
    Volume (4 params): [S1_volume_start, S1_volume_slope, S2_volume_start, S2_volume_slope]
    Bands (12 params): [S1_low_start, S1_low_slope, S1_mid_start, S1_mid_slope, S1_high_start, S1_high_slope,
                        S2_low_start, S2_low_slope, S2_mid_start, S2_mid_slope, S2_high_start, S2_high_slope]
    
    Mask definitions:
    PReLU1(t, st, δt) = min(max(0, (t - st)*δt), 1)

    For volume on S1 (fade-out):
        M_S1_vol(t) = 1 - PReLU1(t, S1_vol_start, S1_vol_slope)

    For volume on S2 (fade-in):
        M_S2_vol(t) = PReLU1(t, S2_vol_start, S2_vol_slope)

    Similarly for bands on S1 (fade-out):
        M_S1_band(t) = 1 - PReLU1(t, S1_band_start, S1_band_slope)

    For bands on S2 (fade-in):
        M_S2_band(t) = PReLU1(t, S2_band_start, S2_band_slope)
    """
    
    # Unpack volume parameters
    S1_vol_start, S1_vol_slope, S2_vol_start, S2_vol_slope = control_signals[:4]
    
    # Unpack band parameters
    band_control_signals = control_signals[4:]
    (S1_low_start, S1_low_slope,
     S1_mid_start, S1_mid_slope,
     S1_high_start, S1_high_slope,
     S2_low_start, S2_low_slope,
     S2_mid_start, S2_mid_slope,
     S2_high_start, S2_high_slope) = band_control_signals

    # Frequency bands
    low_end = n_mels // 3
    mid_end = 2 * n_mels // 3

    frames = S1.shape[1]
    time_vector = torch.arange(frames, dtype=torch.float32, device=S1.device)

    # Volume masks
    # S1 fade-out: M_S1_vol(t) = 1 - PReLU1(t, start, slope)
    S1_vol_mask = 1.0 - prelu1(time_vector, S1_vol_start, S1_vol_slope)
    # S2 fade-in: M_S2_vol(t) = PReLU1(t, start, slope)
    S2_vol_mask = prelu1(time_vector, S2_vol_start, S2_vol_slope)

    # Band masks for S1 (fade-out for each band)
    S1_low_mask = 1.0 - prelu1(time_vector, S1_low_start, S1_low_slope)
    S1_mid_mask = 1.0 - prelu1(time_vector, S1_mid_start, S1_mid_slope)
    S1_high_mask = 1.0 - prelu1(time_vector, S1_high_start, S1_high_slope)

    # Band masks for S2 (fade-in for each band)
    S2_low_mask = prelu1(time_vector, S2_low_start, S2_low_slope)
    S2_mid_mask = prelu1(time_vector, S2_mid_start, S2_mid_slope)
    S2_high_mask = prelu1(time_vector, S2_high_start, S2_high_slope)

    # Construct full masks
    S1_mask = torch.zeros_like(S1)
    S2_mask = torch.zeros_like(S2)

    # For each band, multiply band fade mask by volume mask
    # Low band
    S1_mask[0:low_end, :] = S1_low_mask * S1_vol_mask
    S2_mask[0:low_end, :] = S2_low_mask * S2_vol_mask

    # Mid band
    S1_mask[low_end:mid_end, :] = S1_mid_mask * S1_vol_mask
    S2_mask[low_end:mid_end, :] = S2_mid_mask * S2_vol_mask

    # High band
    S1_mask[mid_end:, :] = S1_high_mask * S1_vol_mask
    S2_mask[mid_end:, :] = S2_high_mask * S2_vol_mask

    # Apply masks to spectrograms
    S_pred = S1 * S1_mask + S2 * S2_mask

    return S_pred

############################################################
# Loss Function
############################################################

def melspectrogram_loss(S_pred, S_truth):
    """
    Define a loss function comparing predicted spectrogram to ground truth.
    A simple L1 or L2 loss can be used initially.
    """
    loss_fn = nn.MSELoss()
    return loss_fn(S_pred, S_truth)

############################################################
# Training Example
############################################################

def beat_time_to_frame(beat_time, bpm_orig, bpm_target, sr, hop_length):
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

# Pseudocode for data loading:
# Assume you have:
#   - S1, S2: raw audio arrays for track 1 and 2
#   - S_truth: mel-spectrogram for the real DJ transition
#   - cue_out_time_s1: the cue-out time (in seconds) for S1
#   - cue_in_time_s2: the cue-in time (in seconds) for S2
#   - bpm_orig_s1, bpm_orig_s2: original BPMs of S1 and S2

# Steps:
# 1) Resample S1 and S2 to TARGET_BPM
# 2) Extract mel-spectrograms
# 3) Compute frame indices for cue points using beat_time_to_resampled_frame
# 4) Extract 15-second segment around the transition
# 5) Prepare input batch and S_truth

def prepare_input_segment(S1_audio_signal, S2_audio_signal, S_truth, 
                          cue_out_time_s1, cue_in_time_s2,
                          bpm_orig_s1, bpm_orig_s2,
                          bpm_target, segment_duration=SEGMENT_DURATION,
                          sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    # Time stretch S1 and S2 to target BPM
    S1_stretched = adjust_bpm(S1_audio_signal, bpm_orig_s1, bpm_target)
    S2_stretched = adjust_bpm(S2_audio_signal, bpm_orig_s2, bpm_target)

    # Convert to mel-spectrograms
    S1_mel = ft.audio_signal_melspectrogram(S1_stretched, sr)
    S2_mel = ft.audio_signal_melspectrogram(S2_stretched, sr)

    # Compute frame indices of cue points after time stretching
    cue_out_frame_s1 = beat_time_to_frame(cue_out_time_s1, bpm_orig_s1, bpm_target, sr, hop_length)
    cue_in_frame_s2 = beat_time_to_frame(cue_in_time_s2, bpm_orig_s2, bpm_target, sr, hop_length)
    
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

    # Handle boundary conditions
    # Pad if needed
    def safe_extract(S, start, end):
        n_mels, n_frames = S.shape
        if start < 0:
            # pad on the left
            left_pad = abs(start)
            S_segment = np.hstack([np.zeros((n_mels, left_pad)), S[:, :max(0, end)]])
        else:
            S_segment = S[:, start:end] if end <= n_frames else np.hstack([S[:, start:], np.zeros((n_mels, end - n_frames))])
        # Ensure exact length
        S_segment = pad_melspectrogram(S_segment, total_frames)
        return S_segment

    S1_segment = safe_extract(S1_mel, start_s1, end_s1)
    S2_segment = safe_extract(S2_mel, start_s2, end_s2)

    # Convert all to torch tensors
    S1_tensor = torch.tensor(S1_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, T)
    S2_tensor = torch.tensor(S2_segment, dtype=torch.float32).unsqueeze(0) # shape: (1, N_MELS, T)
    S_truth_tensor = torch.tensor(S_truth, dtype=torch.float32).unsqueeze(0) # add batch dim

    # Combine S1 and S2 as channels: (batch=1, 2, N_MELS, T)
    input_tensor = torch.cat([S1_tensor, S2_tensor], dim=0).unsqueeze(0)

    return input_tensor, S_truth_tensor


# Example usage in a training loop (simplified):
def train(model, optimizer, train_loader, device='cpu'):
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            # batch should contain input_tensor and S_truth_tensor
            input_tensor, S_truth_tensor = batch
            input_tensor = input_tensor.to(device)
            S_truth_tensor = S_truth_tensor.to(device)

            optimizer.zero_grad()

            # Forward pass
            control_signals = model(input_tensor)  # (batch, volume+band parameters)

            # We need S1 and S2 (as tensors) again to apply masks
            # In a real scenario, you store S1, S2 alongside S_truth in the dataset
            # Here we assume they come together:
            # Let's say your dataset returns (input_tensor, S_truth_tensor, S1_tensor, S2_tensor)
            # For simplicity, assume you have them:
            # S1_tensor, S2_tensor = ...
            # Just placeholders:
            S1_tensor = input_tensor[:,0,:,:]  # shape (batch, N_MELS, T)
            S2_tensor = input_tensor[:,1,:,:]  # shape (batch, N_MELS, T)

            # Apply predicted masks to get S_pred
            # apply_masks expects parameters as a vector; we can do this per example
            # If batch size > 1, loop or vectorize:
            S_pred_list = []
            for i in range(control_signals.size(0)):
                S_pred_example = apply_masks(S1_tensor[i], S2_tensor[i], control_signals[i])
                S_pred_list.append(S_pred_example.unsqueeze(0))
            S_pred = torch.cat(S_pred_list, dim=0)

            # Compute loss
            loss = spectrogram_loss(S_pred, S_truth_tensor)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

############################################################
# Putting it all together:
############################################################

if __name__ == "__main__":
    # Example placeholder code:
    # You have original S1, S2 audio and S_truth spectrogram plus cue points somewhere else in your code.
    S1_audio = np.zeros(int(SAMPLING_RATE*60))  # 60 seconds of silence as placeholder
    S2_audio = np.zeros(int(SAMPLING_RATE*60))  # 60 seconds of silence as placeholder
    S_truth = np.zeros((N_MELS, compute_num_frames(SEGMENT_DURATION)))  # placeholder ground truth spectrogram

    cue_out_time_s1 = 30.0
    cue_in_time_s2 = 35.0
    bpm_orig_s1 = 100.0
    bpm_orig_s2 = 130.0

    # Prepare a single example (in real code, this would be part of a dataset and DataLoader)
    input_tensor, S_truth_tensor = prepare_input_segment(S1_audio, S2_audio, S_truth,
                                                         cue_out_time_s1, cue_in_time_s2,
                                                         bpm_orig_s1, bpm_orig_s2)

    # Instantiate model and optimizer
    model = TransitionPredictor()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dummy dataset
    dataset = [(input_tensor, S_truth_tensor)]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Train
    train(model, optimizer, train_loader, device='cpu')
