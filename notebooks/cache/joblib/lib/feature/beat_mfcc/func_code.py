# first line: 53
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
