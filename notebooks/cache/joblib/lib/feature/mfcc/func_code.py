# first line: 29
@memory.cache
def mfcc(path):
  sig_, sr = librosa.load(path)
  mfcc_ = librosa.feature.mfcc(sig_, sr, n_mfcc=12)
  return mfcc_
