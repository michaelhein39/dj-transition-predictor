# first line: 65
@memory.cache
def chroma_cens(path):
  audio_signal, sr = librosa.load(path)
  chroma_cens_ = librosa.feature.chroma_cens(y=audio_signal, sr=sr)
  return chroma_cens_
