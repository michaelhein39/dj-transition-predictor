# first line: 21
@memory.cache
def beats(path, fps=100):
  beat_processor = BeatTrackingProcessor(fps=fps)
  beat_activations_ = beat_activations(path)
  beats_ = beat_processor(beat_activations_)
  return beats_
