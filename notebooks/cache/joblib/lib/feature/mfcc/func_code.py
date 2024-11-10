# first line: 37
@memory.cache
def mfcc(path):
  """
  Compute MFCC features for an audio signal.
  The output is an array of size (n_mfcc, n_frames).

  n_mfcc: number of MFCCs that the audio is split into
  A higher n_mfcc would be desired for capturing a wider range of spectral features.

  norm: whether to normalize the MFCC features
  """
  audio_signal, sr = librosa.load(path)
  mfcc_ = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=12)
  return mfcc_
