import numpy as np
import librosa
from joblib import Memory
from lib.constants import *

# Monkey patch np.float and np.int (used in Madmom)
np.float = float
np.int = int

# Madmom is generally preferred over Librosa for beat tracking accuracy
# DBNBeatTrackingProcessor is generally preferred over BeatTrackingProcessor
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor

memory = Memory('./cache', verbose=1)

@memory.cache
def beat_activations(path, start_time=None, end_time=None, sr=SAMPLING_RATE):
    """
    RNNBeatProcessor predicts the beat locations in an audio signal.
    The output are the activations, which are the probabilities of each frame
    being a beat (100 frames per second).
    """
    beat_processor = RNNDownBeatProcessor()
    if start_time is not None and end_time is not None:
        y, _ = librosa.load(path, sr=sr, offset=start_time, duration=(end_time - start_time))
        activations = beat_processor(y)
        beat_activations_ = activations[:, 0]
        downbeat_activations_ = activations[:, 1]
    else:
        activations = beat_processor(path)
        beat_activations_ = activations[:, 0]
        downbeat_activations_ = activations[:, 1]
    return beat_activations_, downbeat_activations_


@memory.cache
def beat_times(path, start_time=None, end_time=None, sr=SAMPLING_RATE, fps=FPS):
    """
    DBNBeatTrackingProcessor predicts the beat locations in an audio signal.
    The output is an array of the time stamps (in seconds) of each beat.

    fps: frames per second
    A higher fps will result in more precise beat tracking, but at a higher
    computational cost.
    """
    beat_activations_, _ = beat_activations(path, start_time=start_time, end_time=end_time, sr=sr)
    beattracking_processor = DBNBeatTrackingProcessor(fps=fps)
    return beattracking_processor(beat_activations_)


def audio_signal_stft(audio_signal):
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.

    Returns:
        stft_magnitude (numpy.ndarray): Magnitude of the STFT.
        stft_phase (numpy.ndarray): Phase of the STFT.
    """
    stft_result = librosa.stft(y=audio_signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_magnitude = np.abs(stft_result)
    stft_phase = np.angle(stft_result)
    return stft_result, stft_magnitude, stft_phase


def melspectrogram(path, sr=SAMPLING_RATE):
    # 22050 may be too small for proper alignment and learning,
    # but 44100 may be too big for our computational power

    audio_signal, sr = librosa.load(path, sr=sr)
    melspectrogram_ = librosa.feature.melspectrogram(y=audio_signal, sr=sr,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                     n_mels=N_MELS)
    log_melspectrogram = np.log(melspectrogram_ + 1e-3)
    return log_melspectrogram


def audio_signal_melspectrogram(audio_signal, sr=SAMPLING_RATE):
    melspectrogram_ = librosa.feature.melspectrogram(y=audio_signal, sr=sr,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                     n_mels=N_MELS)
    log_melspectrogram = np.log(melspectrogram_ + 1e-3)
    return log_melspectrogram


def beat_melspectrogram(path):
    beat_times_ = beat_times(path)
    melspectrogram_ = melspectrogram(path)
    beat_melspectrogram_ = beat_aggregate(melspectrogram_, beat_times_)
    return beat_melspectrogram_


@memory.cache
def mfcc(path):
    """
    Compute MFCC features for an audio signal.
    The output is an array of size (n_mfcc, n_frames).

    n_mfcc: number of MFCCs that the audio is split into
    A higher n_mfcc would be desired for capturing a wider range of spectral features.
    """
    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
    mfcc_ = librosa.feature.mfcc(y=audio_signal, sr=sr,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 n_mfcc=N_MFCC)
    return mfcc_


@memory.cache
def beat_mfcc(path):
    """
    Aggregates MFCCs by beat.
    The output is an array of size (n_mfcc, n_beats).
    """
    beat_times_ = beat_times(path)
    mfcc_ = mfcc(path)
    beat_mfcc_ = beat_aggregate(mfcc_, beat_times_)
    return beat_mfcc_


@memory.cache
def chroma_cens(path):
    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
    chroma_cens_ = librosa.feature.chroma_cens(y=audio_signal, sr=sr,
                                              hop_length=HOP_LENGTH,
                                              n_chroma=N_CHROMA)
    return chroma_cens_


@memory.cache
def beat_chroma_cens(path):
    beat_times_ = beat_times(path)
    chroma_cens_ = chroma_cens(path)
    beat_chroma_cens_ = beat_aggregate(chroma_cens_, beat_times_)
    return beat_chroma_cens_


@memory.cache
def chroma_cqt(path):
    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
    chroma_cqt_ = librosa.feature.chroma_cqt(y=audio_signal, sr=sr,
                                            hop_length=HOP_LENGTH,
                                            n_chroma=N_CHROMA)
    return chroma_cqt_


@memory.cache
def beat_chroma_cqt(path):
    beat_times_ = beat_times(path)
    chroma_cqt_ = chroma_cqt(path)
    beat_chroma_cqt_ = beat_aggregate(chroma_cqt_, beat_times_)
    return beat_chroma_cqt_


@memory.cache
def spectral_contrast(path):
    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
    contrast = librosa.feature.spectral_contrast(y=audio_signal, sr=sr,
                                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    return contrast


@memory.cache
def beat_spectral_contrast(path):
    beat_times_ = beat_times(path)
    contrast = spectral_contrast(path)  # Shape is (7, n_beats)
    return beat_aggregate(contrast, beat_times_)


@memory.cache
def onset_strength(path):
    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
    onset_env = librosa.onset.onset_strength(y=audio_signal, sr=sr, hop_length=HOP_LENGTH)
    return onset_env


@memory.cache
def beat_onset_strength(path):
    beat_times_ = beat_times(path)
    onset_env = onset_strength(path)
    onset_env = np.expand_dims(onset_env, axis=0)  # Make it 2D for beat_aggregate
    return beat_aggregate(onset_env, beat_times_)


@memory.cache
def beat_downbeat_probabilities(path):
    """
    Returns downbeat probabilities sampled at beat timestamps.
    This is beat-synchronous (not aggregated).
    """
    # Get beat timestamps using the previously defined beat_times function.
    beat_times_ = beat_times(path)
    
    # Process the audio file with the RNNDownBeatProcessor.
    # The returned activations are assumed to be a 2D array where:
    # - Column 0: beat probabilities.
    # - Column 1: downbeat probabilities.
    _, downbeat_probs = beat_activations(path)
    
    # Convert beat timestamps (seconds) to corresponding frame indices.
    beat_frames = (beat_times_ * FPS).astype(int)  # Same FPS used for RNNDownBeatProcessor
    
    # Clip indices to ensure they don't exceed the array bounds.
    beat_frames = np.clip(beat_frames, 0, len(downbeat_probs) - 1)
    
    # Sample the downbeat probabilities at the beat frames.
    beat_downbeat_probs = downbeat_probs[beat_frames]

    # Make it 2D
    beat_downbeat_probs = np.expand_dims(beat_downbeat_probs, axis=0)
    
    return beat_downbeat_probs


def beat_aggregate(feature, beat_times_):
    """
    Takes a feature of a song and aggregates it by beats so that there is a
    single value for each beat. This allows you to analyze the audio signal
    in terms of beats rather than individual frames.
    The output is an array of size (n_features, n_beats).
    """
    beat_frames = librosa.time_to_frames(beat_times_, sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
    return librosa.util.sync(feature, beat_frames, aggregate=np.mean)