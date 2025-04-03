import torch.nn as nn


VOLUME_CONTROL_SIGNAL_COUNT = 4     # 2 control signals (start, slope) * 2 tracks for volume
BANDPASS_CONTROL_SIGNAL_COUNT = 12  # 2 control signals (start, slope) * 3 bands * 2 tracks


class TransitionPredictor(nn.Module):
    """
    CNN-based model to predict masking parameters from two input mel-spectrograms.
    The input will be a tensor of shape: (batch, 2, N_MELS, T)
    - 2 channels: one for S1, one for S2
    - N_MELS mel bins
    - T frames in the mel-spectrograms (corresponding to 15 seconds)
    """
    def __init__(self,
                 volume_control_signal_count=VOLUME_CONTROL_SIGNAL_COUNT, 
                 bandpass_control_signal_count=BANDPASS_CONTROL_SIGNAL_COUNT):
        super(TransitionPredictor, self).__init__()
        
        # Simple CNN (adjust as needed)
        # Reduces spatial dimensions and extracts features
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

        # Fully connected layers to map features to control signals
        # Total of (volume_control_signal_count + bandpass_control_signal_count) outputs
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, volume_control_signal_count + bandpass_control_signal_count)
        )

        for layer in self.fc.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights with a small standard deviation and biases to zero.
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x shape: (batch, 2, N_MELS, T)
        features = self.conv_layers(x) # shape: (batch, 64, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 64)
        control_signals = self.fc(features)  # (batch, volume_control_signal_count + bandpass_control_signal_count)
        return control_signals