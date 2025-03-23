import os
import pandas as pd
from glob import glob
from collections import namedtuple
from lib.feature import *
from lib.constants import *

# Monkey patch np.float and np.int (used in Madmom)
np.float = float
np.int = int
# Madmom is generally preferred over Librosa for beat tracking accuracy
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

Case = namedtuple('Case', ['features', 'key_invariant'])
CASES = [
  Case(features=['mfcc'], key_invariant=False),
  Case(features=['chroma'], key_invariant=False),
  Case(features=['chroma'], key_invariant=True),
  Case(features=['chroma', 'mfcc'], key_invariant=False),
  Case(features=['chroma', 'mfcc'], key_invariant=True),
]

df_tlist = pd.read_csv('data/meta/tracks_trunc.csv')
df_mlist = pd.read_csv('data/meta/mixes_trunc.csv')

df_align = pd.concat([pd.read_pickle(p) for p in glob('data/align/*.pkl')]).set_index('mix_id')
df_align['case'] = df_align.feature
df_align.loc[df_align.key_invariant, 'case'] += '-keyinv'

# Process transition data frame.
prev = df_tlist.copy()
prev = prev.rename(columns={'i_track': 'i_track_prev'})
prev['i_track_next'] = prev.i_track_prev + 1

next = df_tlist.copy()
next = next.rename(columns={'i_track': 'i_track_next'})
df_trans = prev.merge(next, on='i_track_next', suffixes=('_prev', '_next'))
# df_trans = df_trans[['i_track_prev', 'i_track_next', 'timestamp_prev', 'timestamp_next',
#                      'track_id_prev', 'track_id_next']]


def main():
  os.makedirs('data/segment', exist_ok=True)
  data = []
  for _, mix in df_mlist.iterrows():
    result = segmentation(mix.mix_id)
    data.append(result)
  df = pd.concat(data, ignore_index=True)
  df.to_pickle('data/segment/all_mix_segmentation.pkl')  # Overwrites existing file



def segmentation(mix_id):
  result_path = f'data/segment/{mix_id}.csv'
  if os.path.isfile(result_path):
    print(f'=> Skip processing: {result_path}')
    return pd.read_pickle(result_path)

  df_mix_trans = df_trans
  df_mix_align = df_align
  df_mix_align_prev = df_mix_align.copy()
  df_mix_align_prev.columns = df_mix_align_prev.columns + '_prev'
  df_mix_align_prev = df_mix_align_prev.rename(columns={'case_prev': 'case'})
  df_mix_align_next = df_mix_align.copy()
  df_mix_align_next.columns = df_mix_align_next.columns + '_next'
  df_mix_align_next = df_mix_align_next.rename(columns={'case_next': 'case'})
  df = df_mix_trans.merge(df_mix_align_prev).merge(df_mix_align_next)

  # Each row of df represents a transition between two tracks.
  # It has columns pertaining to the tracklist info of each track, as well as
  # the alignment info between the two tracks.

  mix_path = f'data/mix/{mix_id}.wav'
  mix_beat_times = beat_times(mix_path)

  results = []
  for idx, row in df.iterrows():
    if (row['wp_prev'][-1, 0] != 0) or (row['wp_next'][-1, 0] != 0):
      # Some weird warping path results...
      # I guess the tracks are too long so they are considered as the longer sequence?
      print(f'=> ERROR 1: {mix_id}')
      continue

    if (row['wp_prev'].max() > len(mix_beat_times)) or (row['wp_next'].max() > len(mix_beat_times)):
      # Beats in the warp path should not be greater than the number of beats in the mix
      print(f'=> ERROR 2: {mix_id}')
      continue

    path_S1 = f"data/track/{row['filename_prev']}.wav"
    path_S2 = f"data/track/{row['filename_next']}.wav"

    cue_out_beat_mix = row['mix_cue_out_beat_prev']  # Beginning of transition region
    cue_in_beat_mix = row['mix_cue_in_beat_next']  # End of transition region

    cue_out_time_mix = mix_beat_times[cue_out_beat_mix]  # Beginning of transition region
    cue_in_time_mix = mix_beat_times[cue_in_beat_mix]  # End of transition region

    bpm_orig_S1 = calculate_bpm(path_S1)
    bpm_orig_S2 = calculate_bpm(path_S2)
    bpm_target = calculate_bpm(mix_path,
                               start_time=cue_out_time_mix,
                               end_time=cue_in_time_mix)

    result = {
              'mix_id': mix_id,
              'case': row['case'],
              'i_track_S1': row['i_track_prev'],
              'i_track_S2': row['i_track_next'],
              'track_id_S1': row['track_id_prev'],
              'track_id_S2': row['track_id_next'],
              'cue_out_time_S1': row['track_cue_out_time_prev'],
              'cue_in_time_S2': row['track_cue_in_time_next'],
              'path_S1': path_S1,
              'path_S2': path_S2,
              'mix_path': mix_path,
              'cue_out_time_mix': cue_out_time_mix,
              'cue_in_time_mix': cue_in_time_mix,
              'bpm_orig_S1': bpm_orig_S1,
              'bpm_orig_S2': bpm_orig_S2,
              'bpm_target': bpm_target
              }
    results.append(result)

  # Convert results to DataFrame
  df_results = pd.DataFrame(results)

  # Each row of df_results is a transition between two tracks.
  # The columns include the cue in and cue out for each track as calculated in alignment.py

  df_results.to_csv(result_path)
  print(f'=> Saved: {result_path}')
  return df_results


def calculate_bpm(audio_path, sr=SAMPLING_RATE):
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    return tempo


def calculate_bpm(audio_path, start_time=None, end_time=None, sr=SAMPLING_RATE):
    beat_processor = RNNBeatProcessor()
    beattracking_processor = DBNBeatTrackingProcessor(fps=FPS)
    
    # Load the audio and extract the segment
    if start_time is not None and end_time is not None:
      y, _ = librosa.load(audio_path, sr=sr, offset=start_time, duration=(end_time - start_time))
    else:
      y, _ = librosa.load(audio_path, sr=sr)
    
    beat_activations = beat_processor(y)
    beat_times = beattracking_processor(beat_activations)
    
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        bpm = 60.0 / np.mean(intervals)
    else:
        print(f'=> ERROR: bad segment')
    return bpm


if __name__ == '__main__':
  main()