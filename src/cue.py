import numpy as np

'''
Consider changing the recursive nature of this function to use num_diag values
from a list of possible values, starting from 64 and decreasing linearly by 16.

There is a chance this doesn't really help us much at all though because
there are problems with some of the warp paths due to librosa.
'''


def find_cue(wp, cue_in=False, num_diag=32):
  """
  Args:
    wp: 2D array representing the warping path between two audio files
    cue_in: if True, then output cue-in points, otherwise outputs cue-out points
    num_diag: number of diagonals steps to consider when finding cue point
  Returns:
    (cue point in beats on mix, cue point in beats on track)
  """
  if num_diag == 0:
    if cue_in:
      # Return last beats
      return wp[-1, 1], wp[-1, 0]
    else:
      # Return first beats
      return wp[0, 1], wp[0, 0]

  # Mix component of warp path
  x = wp[::-1, 1]
  dx = np.diff(x)

  # Track component of warp path
  y = wp[::-1, 0]
  dy = np.diff(y)

  with np.errstate(divide='ignore'):
    slope = dy / dx
  slope[np.isinf(slope)] = 0

  if cue_in:
    slope = slope[::-1].cumsum()
    slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]
    slope = slope[::-1]

    # Now the slope array is in chronological order with a cumulative
    # sum of the previous num_diag values applied in reverse order.

    # Get indices where the next num_diag values have a slope of 1.
    # This means that the mix and the track are in sync over the next 
    # num_diag beats.
    i_diag = np.nonzero(slope == num_diag)[0]

    if len(i_diag) == 0:
      # Try again with smaller transition size
      return find_cue(wp, cue_in, num_diag // 2)
    else:
      # Get first index where the mix and track begin to sync
      i = i_diag[0]
      return x[i], y[i]
  else:
    slope = slope.cumsum()
    slope[num_diag:] = slope[num_diag:] - slope[:-num_diag]

    # Now the slope array is in chronological order with a cumulative
    # sum of the previous num_diag values applied in order.

    # Get indices where the past num_diag values have a slope of 1.
    # This means that the mix and the track are in sync over the past 
    # num_diag beats.
    i_diag = np.nonzero(slope == num_diag)[0]

    if len(i_diag) == 0:
      # Try again with smaller transition size
      return find_cue(wp, cue_in, num_diag // 2)
    else:
      # Get last index where the mix and track are in sync
      i = i_diag[-1]
      return x[i] + 1, y[i] + 1
