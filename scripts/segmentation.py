import os
import pandas as pd
from glob import glob
from collections import namedtuple
from lib.feature import *
from lib.constants import *

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
  result_path = f'data/segment/{mix_id}.pkl'
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

  for idx, row in df.iterrows():
    if (row.wp_prev[-1, 0] != 0) or (row.wp_next[-1, 0] != 0):
      # Some weird warping path results...
      # I guess the tracks are too long so they are considered as the longer sequence?
      print(f'=> ERROR 1: {mix_id}')
      continue

    if (row.wp_prev.max() > len(mix_beat_times)) or (row.wp_next.max() > len(mix_beat_times)):
      # Beats in the warp path should not be greater than the number of beats in the mix
      print(f'=> ERROR 2: {mix_id}')
      continue

    cue_out_beat_mix = row.mix_cue_out_beat_prev
    cue_in_beat_mix = row.mix_cue_in_beat_next

    cue_out_time_mix = mix_beat_times[cue_out_beat_mix]
    cue_in_time_mix = mix_beat_times[cue_in_beat_mix]

  df = df[['case', 'i_track_prev', 'i_track_next', 'track_id_prev', 'track_id_next',
           'match_rate_prev', 'match_rate_next',
           'mix_cue_out_time', 'mix_cue_in_time', 'mix_cue_mid_time',
           'mix_cue_out_beat', 'mix_cue_in_beat', 'mix_cue_mid_beat',
           'track_cue_in_time_prev', 'track_cue_out_time_prev',
           'track_cue_in_time_next', 'track_cue_out_time_next',
           'track_cue_in_beat_prev', 'track_cue_out_beat_prev',
           'track_cue_in_beat_next', 'track_cue_out_beat_next',
           'key_change_prev', 'key_change_next',
           'wp_prev', 'wp_next',
           ]]
  df['mix_id'] = mix_id

  result = {
            'case': row['case'],
            'i_track_S1': row['i_track_prev'],
            'i_track_S2': row['i_track_next'],
            'track_id_S1': row['track_id_prev'],
            'track_id_S2': row['track_id_next'],
            'match_rate_S1': row['match_rate_prev'],
            'match_rate_S2': row['match_rate_next'],
            'track_cue_in_time_prev': row['track_cue_in_time_prev'],
            'cue_out_time_S1': row['track_cue_out_time_prev'],
            'cue_in_time_S2': row['track_cue_in_time_next'],
            'track_cue_out_time_next': row['track_cue_out_time_next'],
            'path_S1': path_S1,
            'path_S2': path_S2,
            'mix_path': mix_path,
            'cue_out_time_mix': mix_cue_in_time,
            'cue_in_time_mix': mix_cue_out_time,
            'bpm_orig_S1': bpm_orig_S1,
            'bpm_orig_S2': bpm_orig_S2,
            'bpm_target': bpm_target
        }
  # Convert results to DataFrame
  df = pd.DataFrame(result)

  # Each row of df is a transition between two tracks.
  # The columns include the cue in and cue out for each track as calculated
  # in alignment.py, as well as the cue points based on tracklist timestamp metadata.

  df.to_pickle(result_path)
  print(f'=> Saved: {result_path}')
  return df


def calculate_bpm(audio_path):
    y, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
    tempo, _ = librosa.beat.beat_track(y, sr=SAMPLING_RATE)
    return tempo


if __name__ == '__main__':
  main()