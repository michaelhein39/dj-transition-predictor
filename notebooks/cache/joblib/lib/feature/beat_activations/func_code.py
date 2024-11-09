# first line: 9
@memory.cache
def beat_activations(path):
  """
  RNNBeatProcessor predicts the beat locations in an audio signal.
  The output are the activations, which are the probabilities of each time step
  being a beat.
  """
  beat_processor = RNNBeatProcessor()
  beat_activations_ = beat_processor(path)
  return beat_activations_
