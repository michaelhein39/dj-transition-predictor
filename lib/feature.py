import numpy as np
import librosa
from joblib import Memory
from lib.constants import *

# Monkey patch np.float and np.int (used in Madmom)
np.float = float
np.int = int
# Madmom is generally preferred over Librosa for beat tracking accuracy
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor

memory = Memory('./cache', verbose=1)

@memory.cache
def beat_activations(path):
  """
  RNNBeatProcessor predicts the beat locations in an audio signal.
  The output are the activations, which are the probabilities of each frame
  being a beat (100 frames per second).
  """
  beat_processor = RNNBeatProcessor()
  beat_activations_ = beat_processor(path)
  return beat_activations_


@memory.cache
def beat_times(path, fps=FPS):
  """
  BeatTrackingProcessor predicts the beat locations in an audio signal.
  The output is an array of the time stamps (in seconds) of each beat.

  fps: frames per second
  A higher fps will result in more precise beat tracking, but at a higher
  computational cost.
  """
  beat_activations_ = beat_activations(path)
  beat_processor = BeatTrackingProcessor(fps=fps)
  beat_times_ = beat_processor(beat_activations_)
  return beat_times_


def melspectrogram(path):
    # 22050 may be too small for proper alignment and learning,
    # but 44100 may be too big for our computational power

    audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
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


def beat_aggregate(feature, beat_times_, frames_per_beat=None):
  """
  Takes a feature of a song and aggregates it by beats so that there is a
  single value for each beat. This allows you to analyze the audio signal
  in terms of beats rather than individual frames.
  The output is an array of size (n_features, n_beats).
  """
  beat_frames = librosa.time_to_frames(beat_times_, sr=SAMPLING_RATE, hop_length=HOP_LENGTH)
  return librosa.util.sync(feature, beat_frames, aggregate=np.mean)
