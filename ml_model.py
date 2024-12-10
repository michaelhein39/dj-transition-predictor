import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa

############################################################
# Constants and Hyperparameters
############################################################
SAMPLING_RATE = 22050       # Sampling rate for audio
HOP_LENGTH = 512            # Hop length for STFT -> mel-spectrogram
N_MELS = 128                # Number of mel-frequency bins
TARGET_BPM = 120.0          # Example target BPM for normalization
SEGMENT_DURATION = 15.0     # 15 seconds segment length
FADE_PARAMETER_COUNT = 4    # 2 parameters (start, slope) * 2 tracks for volume
BANDPASS_PARAMETER_COUNT = 12 # 2 parameters (start, slope) * 3 bands * 2 tracks

# You may adjust these as needed based on your network design:
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
EPOCHS = 10

############################################################
# Utility Functions
############################################################

def resample_to_target_bpm(audio, sr, bpm_orig, bpm_target=TARGET_BPM):
    """
    Resample audio to match the target BPM.
    This involves time-stretching the audio so that its tempo matches bpm_target.
    """
    # Calculate time-stretch factor:
    # if original BPM is bpm_orig and target BPM is bpm_target,
    # factor = bpm_orig / bpm_target. If bpm_orig = 100 and bpm_target = 120,
    # factor would be 100/120 = 0.8333, meaning we need to speed up slightly.
    time_stretch_factor = bpm_orig / bpm_target
    # Use librosa's time_stretch (phase vocoder) to match BPM
    # Note: librosa.time_stretch expects the STFT or the raw time series?
    # For raw audio, first we could transform to time-freq domain, then apply time stretch,
    # but librosa.time_stretch works directly on waveforms.
    audio_stretched = librosa.effects.time_stretch(audio, rate=1.0/time_stretch_factor)
    return audio_stretched, sr

def audio_to_mel_spectrogram(audio, sr, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """
    Convert raw audio to a mel-spectrogram.
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def pad_spectrogram_to_length(S, target_frames):
    """
    Pad or truncate a mel-spectrogram S (shape: n_mels x frames) to have exactly target_frames in time dimension.
    If S is shorter, pad with zeros. If S is longer, truncate.
    """
    n_mels, frames = S.shape
    if frames < target_frames:
        # Pad on the right side with zeros
        pad_amount = target_frames - frames
        S_padded = np.hstack([S, np.zeros((n_mels, pad_amount))])
    else:
        # Truncate if longer than needed (should be rare if we carefully extracted)
        S_padded = S[:, :target_frames]
    return S_padded

def compute_num_frames(duration_sec, sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    """
    Given a duration in seconds, compute how many STFT frames correspond to that duration.
    Number of frames = floor(duration_sec * sr / hop_length)
    """
    frames = int(np.floor((duration_sec * sr) / hop_length))
    return frames

############################################################
# Model Definition
############################################################

class TransitionPredictor(nn.Module):
    """
    CNN-based model to predict masking parameters from two input mel-spectrograms.
    The input will be a tensor of shape: (batch, 2, N_MELS, T)
    - 2 channels: one for S1, one for S2
    - N_MELS mel bins
    - T frames corresponding to 15 seconds
    """
    def __init__(self, n_mels=N_MELS, time_frames=1000,  # time_frames is a placeholder; we will compute it dynamically
                 volume_param_count=FADE_PARAMETER_COUNT, 
                 band_param_count=BANDPASS_PARAMETER_COUNT):
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
        # We have total of volume_param_count + band_param_count parameters
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, volume_param_count + band_param_count)
        )

    def forward(self, x):
        # x shape: (batch, 2, N_MELS, T)
        features = self.conv_layers(x) # shape: (batch, 64, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 64)
        params = self.fc(features)  # (batch, volume_param_count + band_param_count)
        return params

############################################################
# Masking Functions
############################################################

def apply_masks(S1, S2, params, n_mels=N_MELS):
    """
    Given the predicted parameters and the original S1 and S2 spectrograms,
    apply the volume and band-pass masks to produce the predicted transition spectrogram.
    
    params should contain:
    - volume parameters: [S1_volume_start, S1_volume_slope, S2_volume_start, S2_volume_slope]
    - band parameters: 12 values corresponding to (low, mid, high) * (start, slope) for S1 and S2
    
    For simplicity, let's just show how we might construct linear ramps.
    In practice, you will need to define how to segment the frequency bands and apply slopes.
    """
    # Parse parameters
    # volume params (2 per track)
    S1_vol_start, S1_vol_slope, S2_vol_start, S2_vol_slope = params[:4]
    
    # band params (6 per track: 3 bands * 2 params each)
    # Let's assume order: (S1_low_start, S1_low_slope, S1_mid_start, S1_mid_slope, S1_high_start, S1_high_slope,
    #                      S2_low_start, S2_low_slope, S2_mid_start, S2_mid_slope, S2_high_start, S2_high_slope)
    band_params = params[4:]
    
    # For demonstration, create a time vector:
    frames = S1.shape[1]
    time_vector = torch.arange(frames, dtype=torch.float32, device=S1.device)

    # Volume masks (linear ramps)
    S1_vol_mask = (time_vector - S1_vol_start) * S1_vol_slope
    S2_vol_mask = (time_vector - S2_vol_start) * S2_vol_slope

    # Clip values so that volume stays between 0 and 1
    S1_vol_mask = torch.clamp(S1_vol_mask, min=0.0, max=1.0)
    S2_vol_mask = torch.clamp(S2_vol_mask, min=0.0, max=1.0)

    # Define frequency band splits. For simplicity:
    # low band: 0 to N_MELS//3
    # mid band: N_MELS//3 to 2*N_MELS//3
    # high band: 2*N_MELS//3 to N_MELS
    low_end = n_mels // 3
    mid_end = 2 * n_mels // 3

    # Extract band parameters
    (S1_low_start, S1_low_slope,
     S1_mid_start, S1_mid_slope,
     S1_high_start, S1_high_slope,
     S2_low_start, S2_low_slope,
     S2_mid_start, S2_mid_slope,
     S2_high_start, S2_high_slope) = band_params

    # Construct band masks similarly (linear ramps)
    S1_low_mask = torch.clamp((time_vector - S1_low_start)*S1_low_slope, 0, 1)
    S1_mid_mask = torch.clamp((time_vector - S1_mid_start)*S1_mid_slope, 0, 1)
    S1_high_mask = torch.clamp((time_vector - S1_high_start)*S1_high_slope, 0, 1)

    S2_low_mask = torch.clamp((time_vector - S2_low_start)*S2_low_slope, 0, 1)
    S2_mid_mask = torch.clamp((time_vector - S2_mid_start)*S2_mid_slope, 0, 1)
    S2_high_mask = torch.clamp((time_vector - S2_high_start)*S2_high_slope, 0, 1)

    # Construct full frequency-time mask for each track
    S1_mask = torch.zeros_like(S1)
    S2_mask = torch.zeros_like(S2)

    # Apply band masks and volume masks multiplicatively
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

def spectrogram_loss(S_pred, S_truth):
    """
    Define a loss function comparing predicted spectrogram to ground truth.
    A simple L1 or L2 loss can be used initially.
    """
    loss_fn = nn.MSELoss()
    return loss_fn(S_pred, S_truth)

############################################################
# Training Example
############################################################

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

def prepare_input_segment(S1_audio, S2_audio, S_truth, 
                          cue_out_time_s1, cue_in_time_s2,
                          bpm_orig_s1, bpm_orig_s2,
                          bpm_target=TARGET_BPM, sr=SAMPLING_RATE, hop_length=HOP_LENGTH):
    # Resample S1 and S2 to target BPM
    S1_resampled, sr = resample_to_target_bpm(S1_audio, sr, bpm_orig_s1, bpm_target)
    S2_resampled, sr = resample_to_target_bpm(S2_audio, sr, bpm_orig_s2, bpm_target)

    # Convert to mel-spectrograms
    S1_mel = audio_to_mel_spectrogram(S1_resampled, sr)
    S2_mel = audio_to_mel_spectrogram(S2_resampled, sr)

    # Compute frame indices of cue points after resampling
    # You have a function beat_time_to_resampled_frame provided in your codebase:
    cue_out_frame_s1 = beat_time_to_resampled_frame(cue_out_time_s1, bpm_orig_s1, bpm_target, sr, hop_length)
    cue_in_frame_s2 = beat_time_to_resampled_frame(cue_in_time_s2, bpm_orig_s2, bpm_target, sr, hop_length)
    
    # Determine how many frames correspond to 15 seconds
    total_frames = compute_num_frames(SEGMENT_DURATION, sr, hop_length) 
    
    # Suppose we know z = cue_in_frame_s2 - cue_out_frame_s1 is the overlap region
    # If z < total_frames, we need to add (x seconds before and y seconds after)
    # For simplicity, assume we have z_frames = some calculation based on S_truth length:
    z_frames = S_truth.shape[1]  # ground truth transition frames
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
    # S1 segment: start from cue_out_frame_s1 - x_frames and go until cue_out_frame_s1 + z_frames + y_frames
    # But we only need 15 seconds total. So total_frames = x_frames + z_frames + y_frames.
    start_s1 = cue_out_frame_s1 - x_frames
    end_s1 = start_s1 + total_frames
    start_s2 = (cue_in_frame_s2 - (total_frames - y_frames)) # (15 - y) seconds before cue_in
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
        S_segment = pad_spectrogram_to_length(S_segment, total_frames)
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
            params = model(input_tensor)  # (batch, volume+band parameters)

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
            for i in range(params.size(0)):
                S_pred_example = apply_masks(S1_tensor[i], S2_tensor[i], params[i])
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
