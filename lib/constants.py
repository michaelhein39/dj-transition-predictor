SAMPLING_RATE = 22050  # 44100 is the min in the trunc sample, but we use 22050
N_FFT = 2048
HOP_LENGTH = 512  # Hop length for STFT -> mel-spectrogram
N_MELS = 128
N_MFCC = 12
N_CHROMA = 12
FPS = 100
SEGMENT_DURATION = 30  # 15 seconds segment length