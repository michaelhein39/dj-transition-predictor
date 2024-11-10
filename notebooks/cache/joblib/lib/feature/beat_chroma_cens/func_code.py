# first line: 72
@memory.cache
def beat_chroma_cens(path):
  beat_times_ = beat_times(path)
  chroma_cens_ = chroma_cens(path)
  beat_chroma_cens_ = beat_aggregate(chroma_cens_, beat_times_)
  return beat_chroma_cens_
