import numpy as np
import librosa
from joblib import Memory

# Monkey patch np.float and np.int (used in madmom)
np.float = float
np.int = int

from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor

SAMPLING_RATE = 22050  # 44100 is the min in the trunc sample, but we use 22050
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
def beat_times(path, fps=100):
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
                                                     n_fft=2048, hop_length=512,
                                                     n_mels=128)
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

  norm: whether to normalize the MFCC features
  """
  audio_signal, sr = librosa.load(path, sr=SAMPLING_RATE)
  mfcc_ = librosa.feature.mfcc(y=audio_signal, sr=sr,
                               n_fft=2048, hop_length=512,
                               n_mfcc=12)
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
                                             hop_length=512,
                                             n_chroma=12)
  return chroma_cens_


@memory.cache
def beat_chroma_cens(path):
  beat_times_ = beat_times(path)
  chroma_cens_ = chroma_cens(path)
  beat_chroma_cens_ = beat_aggregate(chroma_cens_, beat_times_)
  return beat_chroma_cens_


def beat_aggregate(feature, beat_times_, frames_per_beat=None):
  """
  Takes a feature of a song and "groups" it by beats so that there is a
  single value for each beat. This allows you to analyze the audio signal
  in terms of beats rather than individual time stamps.
  The output is an array of size (n_features, n_beats).
  """
  max_frame = feature.shape[1]
  beat_frames = librosa.time_to_frames(beat_times_)
  beat_frames = beat_frames[beat_frames < max_frame]
  beat_feature = np.split(feature, beat_frames, axis=1)

  # Average for each beat.
  beat_feature = beat_feature[1:-1]  # only use chroma features between beats. not before or after beat
  beat_feature = [f.mean(axis=1) for f in beat_feature]  # average chroma features for each beat
  beat_feature = np.array(beat_feature).T
  return beat_feature
