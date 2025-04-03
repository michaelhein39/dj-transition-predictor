import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def train_model(model, 
                train_loader, 
                optimizer,
                model_save_name,
                seed,
                epochs=10, 
                device='cpu',
                save_dir='checkpoints',
                update_interval=20):
    """
    Train the TransitionPredictor model using spectrogram masking and an MSE loss.
    This version expects the DataLoader to yield only (input_tensor, S_truth_tensor),
    where input_tensor has shape (batch, 2, N_MELS, T).
    We slice out S1 and S2 from the input tensor inside the loop.
    
    Args:
        model (nn.Module): The TransitionPredictor (or equivalent) model.
        train_loader (DataLoader or iterable): Yields batches of 
            (input_tensor, S_truth_tensor).
            - input_tensor: shape (batch, 2, N_MELS, T)
            - S_truth_tensor: shape (batch, N_MELS, T)
        optimizer (torch.optim.Optimizer): Optimizer (e.g. Adam) for model parameters.
        epochs (int): Number of epochs to train.
        device (str): 'cpu' or 'cuda'.
        
    Returns:
        None: The model is trained in-place. Prints loss after each epoch.
    """

    # Set the seed for reproducibility
    set_seed(seed)

    # Move model to the specified device (GPU or CPU)
    model.to(device)
    model.train()

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    loss_array = []

    # Loop over epochs
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create a progress bar for the training loop
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
            # Iterate over batches from the train_loader
            for batch_idx, batch_data in enumerate(train_loader):
                # Expecting batch_data = (input_tensor, S_truth_tensor)
                input_tensor, S_truth_tensor = batch_data
                
                # Move data to device
                input_tensor = input_tensor.to(device)     # shape: (batch, 2, N_MELS, T)
                S_truth_tensor = S_truth_tensor.to(device) # shape: (batch, N_MELS, T)

                # Reset gradients from previous iteration
                optimizer.zero_grad()
                
                # Forward pass: model predicts the control signals
                # control_signals shape: (batch, volume_control_signal_count + bandpass_control_signal_count)
                control_signals = model(input_tensor)
                
                # We'll accumulate each example's predicted spectrogram, then stack them.
                batch_size = control_signals.size(0)
                S_pred_list = []
                
                for i in range(batch_size):
                    # Extract the control signals for the i-th item in the batch
                    # shape: (volume_control_signal_count + bandpass_control_signal_count,)
                    ctrl_signals_i = control_signals[i]

                    # Slice the i-th example from input_tensor => shape: (2, N_MELS, T)
                    # Then extract the channels for S1 and S2
                    S_i = input_tensor[i]    # shape: (2, N_MELS, T)
                    S1_i = S_i[0]           # shape: (N_MELS, T)
                    S2_i = S_i[1]           # shape: (N_MELS, T)

                    # apply_masks returns a tensor of shape (N_MELS, T)
                    S_pred_i = apply_masks(S1_i, S2_i, ctrl_signals_i, n_mels=S1_i.size(0))
                    S_pred_list.append(S_pred_i.unsqueeze(0))
                
                # Stack all predicted spectrograms in the batch
                # final shape: (batch, N_MELS, T)
                S_pred = torch.cat(S_pred_list, dim=0)

                # Compute loss against ground truth
                loss = mse_loss(S_pred, S_truth_tensor)
                
                # Backpropagation
                loss.backward()

                 # Check for meaningful gradient norms
                if (batch_idx + 1) % update_interval == 0:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"Step {batch_idx} | Loss: {loss.item():.4f} | Grad Norm: {total_norm:.4f}")

                # Gradient update
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += loss.item()
                loss_array.append(loss.item())
                num_batches += 1

                # Update the progress bar every `update_interval` batches
                if (batch_idx + 1) % update_interval == 0:
                    pbar.update(update_interval)
                    pbar.set_postfix({"batch_loss": loss.item()})

            # Update any remaining batches
            if num_batches % update_interval != 0:
                pbar.update(num_batches % update_interval)
            
        # Print average loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.6f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f'{model_save_name}_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(save_dir, f'{model_save_name}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    return loss_array


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


############################################################
# Loss Functions
############################################################

def mse_loss(S_pred, S_truth):
    """
    MSE loss (L2) between predicted and ground truth mel-spectrograms.
    """
    loss_fn = nn.MSELoss(reduction='sum')
    return loss_fn(S_pred, S_truth)


def l1_loss(S_pred, S_truth):
    """
    L1 loss (Mean Absolute Error) between predicted and ground truth mel-spectrograms.
    """
    loss_fn = nn.L1Loss()
    return loss_fn(S_pred, S_truth)


def spectral_hybrid_loss(S_pred, S_truth, lambda_sc=0.1):
    """
    Combined loss: L1 loss plus spectral convergence loss.
    
    Spectral convergence loss is defined as the ratio of the Frobenius norm of 
    the difference (S_truth - S_pred) to the Frobenius norm of S_truth.
    
    Args:
        S_pred (Tensor): Predicted mel-spectrogram, shape (batch, N_MELS, T).
        S_truth (Tensor): Ground truth mel-spectrogram, shape (batch, N_MELS, T).
        lambda_sc (float): Weight for the spectral convergence term.
        
    Returns:
        Tensor: The combined loss.
    """
    # L1 loss (Mean Absolute Error)
    l1_loss = nn.L1Loss()(S_pred, S_truth)
    
    # Spectral convergence loss
    # Compute the Frobenius norm (over the mel and time dimensions) for each example.
    diff_norm = torch.norm(S_truth - S_pred, p='fro', dim=[1, 2])
    truth_norm = torch.norm(S_truth, p='fro', dim=[1, 2])
    spectral_conv_loss = torch.mean(diff_norm / truth_norm)
    
    return l1_loss + lambda_sc * spectral_conv_loss

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


def reparameterize_fade(raw_st, raw_delta, frames, epsilon=1e-6, beta=0.01):
    # Reparameterize st to be in [0, frames]
    st = frames * torch.sigmoid(raw_st)  # with proper initialization, sigmoid(0) = 0.5, so st starts at frames/2

    # Reparameterize delta using softplus and a scaling factor
    # Lower bound: 1/(Cout - st) ensures the fade doesn't fully complete too early.
    delta = beta * F.softplus(raw_delta) + (1.0 / (frames - st + epsilon))
    return st, delta


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
    raw_S1_vol_start, raw_S1_vol_slope, raw_S2_vol_start, raw_S2_vol_slope = control_signals[:4]
    
    # Unpack band parameters
    band_control_signals = control_signals[4:]
    (raw_S1_low_start, raw_S1_low_slope,
     raw_S1_mid_start, raw_S1_mid_slope,
     raw_S1_high_start, raw_S1_high_slope,
     raw_S2_low_start, raw_S2_low_slope,
     raw_S2_mid_start, raw_S2_mid_slope,
     raw_S2_high_start, raw_S2_high_slope) = band_control_signals

    # Frequency bands
    low_end = n_mels // 3
    mid_end = 2 * n_mels // 3

    frames = S1.shape[1]
    time_vector = torch.arange(frames, dtype=torch.float32, device=S1.device)

    # Reparameterize fade
    S1_vol_start, S1_vol_slope = reparameterize_fade(raw_S1_vol_start, raw_S1_vol_slope, frames)
    S2_vol_start, S2_vol_slope = reparameterize_fade(raw_S2_vol_start, raw_S2_vol_slope, frames)

    S1_low_start, S1_low_slope = reparameterize_fade(raw_S1_low_start, raw_S1_low_slope, frames)
    S1_mid_start, S1_mid_slope = reparameterize_fade(raw_S1_mid_start, raw_S1_mid_slope, frames)
    S1_high_start, S1_high_slope = reparameterize_fade(raw_S1_high_start, raw_S1_high_slope, frames)

    S2_low_start, S2_low_slope = reparameterize_fade(raw_S2_low_start, raw_S2_low_slope, frames)
    S2_mid_start, S2_mid_slope = reparameterize_fade(raw_S2_mid_start, raw_S2_mid_slope, frames)
    S2_high_start, S2_high_slope = reparameterize_fade(raw_S2_high_start, raw_S2_high_slope, frames)

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