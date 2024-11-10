# first line: 21
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
