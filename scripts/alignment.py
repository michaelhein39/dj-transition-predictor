import os
import librosa
import numpy as np
import pandas as pd
import lib.feature as ft
from src.cue import find_cue
from collections import namedtuple


Case = namedtuple('Case', ['features', 'key_invariant'])
CASES = [
    Case(features=['mfcc'], key_invariant=False),
    Case(features=['chroma'], key_invariant=False),
    Case(features=['chroma'], key_invariant=True),
    Case(features=['spectral_contrast'], key_invariant=False),
    Case(features=['chroma', 'mfcc'], key_invariant=False),
    Case(features=['chroma', 'mfcc'], key_invariant=True),
    Case(features=['chroma', 'mfcc', 'spectral_contrast'], key_invariant=False),
    Case(features=['chroma', 'mfcc', 'spectral_contrast', 'downbeat_prob', 'onset_strength'], key_invariant=False),
]


df_tlist = pd.read_csv('data/meta/tracks_trunc.csv')
df_mlist = pd.read_csv('data/meta/mixes_trunc.csv')


def main():
    os.makedirs('data/align', exist_ok=True)
    for _, mix in df_mlist.iterrows():
        # for case in CASES:
        #     alignment(mix.mix_id, features=case.features, key_invariant=case.key_invariant)
        alignment(mix.mix_id,
                  features=['chroma', 'mfcc', 'spectral_contrast', 'downbeat_prob', 'onset_strength'],
                  key_invariant=False)
        alignment(mix.mix_id, features=['spectral_contrast'], key_invariant=False)
        alignment(mix.mix_id, features=['chroma', 'mfcc', 'spectral_contrast'], key_invariant=False)


def alignment(mix_id, features=['chroma', 'mfcc'], key_invariant=True):
    feature_id = feature = '+'.join(features)
    result_path = f'data/align/{mix_id}-{feature_id}'
    if key_invariant:
        result_path += '-key_invariant'
    result_path += '.pkl'
    if os.path.isfile(result_path):
        print(f'=> Skip processing: {result_path}')
        return pd.read_pickle(result_path)

    mix_path = f'data/mix/{mix_id}.wav'
    df_tlist_curr = df_tlist[df_tlist['mix_id'] == mix_id]

    # Feature values at each beat
    # 2D matrix with beats on x-axis and different feature at each y-value
    _ = extract_feature('data/track/jwmtj61_29_1r9f2y1p.wav', features)
    mix_feature = extract_feature(mix_path, features)

    # Beat times
    # 1D vector
    mix_beat_times = ft.beat_times(mix_path)

    data = []
    for _, track in df_tlist_curr.iterrows():
        track_path = f'data/track/{track.filename}.wav'

        # Skip downloading if file already exists
        if not os.path.exists(track_path):
            print(f"File does not exist: {track_path}")
            continue

        track_feature = extract_feature(track_path, features)

        pitch_shifts = np.arange(12) if key_invariant else [0]
        best_cost = np.inf
        best_key_change = np.nan
        best_wp = None
        costs = []

        for pitch_shift in pitch_shifts:
            if pitch_shift == 0:
                X, Y = track_feature, mix_feature
            else:
                X, Y = track_feature.copy(), mix_feature.copy()

                # Circular pitch shifting only the chroma features (12 is the # of bins)
                # We do not pitch shift the mfcc features because they have no direct relationship
                # with musical notes
                X[:12] = np.roll(X[:12], pitch_shift, axis=0)

            # Subsequence DTW.
            # Subsequence tag is used because only a subsequence of the mix is
            # aligned with each track and not the whole mix (used to improve accuracy).
            # D will have shape (track_feature_length, mix_feature_length).
            # wp has shape (L, 2), where L is the length of the warping path, and each row contains
            # the indices of the track feature and the mix feature that are aligned at each point in the warping path.
            D, wp = librosa.sequence.dtw(X, Y, subseq=True)

            # wp will contain indices for the entire track, including parts that are not actually 
            # used in the mix. So it's length will be the number of beats in the track, not the mix.

            # Calculate the matching function, which represents the cumulative distance between
            # the track feature and the mix feature at each point in time, normalized by the length 
            # of the warping path. This allows us to evaluate the quality of the match at each point in time.
            matching_function = D[-1, :] / wp.shape[0]

            # Calculate the cost as the minimum value of the matching function, representing the 
            # point in time where the track feature and the mix feature are best aligned.
            cost = matching_function.min()

            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_key_change = pitch_shift
                best_wp = wp

        # Compute cue_points
        track_beats = ft.beat_times(track_path)
        mix_cue_in_beat, track_cue_in_beat = find_cue(best_wp, cue_in=True)
        mix_cue_out_beat, track_cue_out_beat = find_cue(best_wp, cue_in=False)
        mix_cue_in_time, track_cue_in_time = mix_beat_times[mix_cue_in_beat], track_beats[track_cue_in_beat]
        mix_cue_out_time, track_cue_out_time = mix_beat_times[mix_cue_out_beat], track_beats[track_cue_out_beat]

        # Reverse the warp path to make it easier to find the mix cues
        best_wp_sliced = best_wp[::-1]

        # Find the indices of the mix cues in the reversed warp path
        mix_cue_in_idx = np.where(best_wp_sliced[:, 1] == mix_cue_in_beat)[0][-1]  # Last instance
        mix_cue_out_idx = np.where(best_wp_sliced[:, 1] == mix_cue_out_beat)[0][0]  # First instance

        # Slice the warp path to include only the region between the mix cues
        # Use mix cues rather than track cues because we want to evaluate how well the track is
        # aligned to the mix, not the other way around. Also the mix cues seem to be more accurate
        # than the track cues based on observation.
        best_wp_sliced = best_wp_sliced[mix_cue_in_idx : mix_cue_out_idx + 1]

        # Compute the match rate between the track and mix features
        # Get the mix feature indices and track feature indices from the sliced warp path
        x = best_wp_sliced[:, 1]  # Mix feature beats
        y = best_wp_sliced[:, 0]  # Track feature beats

        # Compute the differences in the track and mix feature indices
        dydx = np.diff(y) / np.diff(x)

        # Compute the match rate as the proportion of differences that are equal to 1
        match_rate = (dydx == 1).sum() / len(dydx)

        # The match rate indicates how well the track and mix features are synchronized,
        # with a value of 1 indicating perfect synchronization and a value close to 0 
        # indicating poor synchronization.

        data.append((
            mix_id, track.track_id, feature, key_invariant,
            match_rate, best_key_change, best_cost, costs, best_wp, best_wp_sliced,
            mix_cue_in_time, mix_cue_out_time, track_cue_in_time, track_cue_out_time,
            mix_cue_in_beat, mix_cue_out_beat, track_cue_in_beat, track_cue_out_beat,
        ))

    df_result = pd.DataFrame(data, columns=[
        'mix_id', 'track_id', 'feature', 'key_invariant',
        'match_rate', 'key_change', 'best_cost', 'costs', 'wp', 'wp_sliced',
        'mix_cue_in_time', 'mix_cue_out_time', 'track_cue_in_time', 'track_cue_out_time',
        'mix_cue_in_beat', 'mix_cue_out_beat', 'track_cue_in_beat', 'track_cue_out_beat',
    ])
    df_result.to_pickle(result_path)
    print(f'=> Saved: {result_path}')
    return df_result


def extract_feature(path, feature_names):
    combined_feature = []
    feature_shapes = {}

    for feature_name in feature_names:
        # Calculate the feature
        if feature_name == 'chroma':
            f = ft.beat_chroma_cens(path).astype('float32')
        elif feature_name == 'mfcc':
            f = ft.beat_mfcc(path).astype('float32')
        elif feature_name == 'spectral_contrast':
            f = ft.beat_spectral_contrast(path).astype('float32')
        elif feature_name == 'downbeat_prob':
            # Beat-synchronous instead of beat-aggregated
            f = ft.beat_downbeat_probabilities(path).astype('float32')
        elif feature_name == 'onset_strength':
            f = ft.beat_onset_strength(path).astype('float32')
        else:
            raise Exception(f'Unknown feature: {feature_name}')
        
        # Normalize the feature
        f = (f - f.mean()) / f.std()

        # Apply weighting for low-dimensional features
        if feature_name in ['downbeat_prob', 'onset_strength']:
            # Equivalent to replicating 6 times
            # This is done to collectively mimic the size of MFCC and CENS
            scale_factor = np.sqrt(6)
            f = f * scale_factor

        feature_shapes[feature_name] = f.shape[-1]
        combined_feature.append(f)

    # Check beat dimensions
    beat_dims = list(feature_shapes.values())
    min_beats = min(beat_dims)
    max_beats = max(beat_dims)
    if max_beats - min_beats > 1:
        raise ValueError("Beat dimensions differ by more than 1 among features")
    elif max_beats - min_beats == 1:
        # Check that only downbeat_prob has the lower beat count
        for name, dim in feature_shapes.items():
            if name == 'downbeat_prob':
                if dim != min_beats:
                    raise ValueError("downbeat_prob should have one less beat than the others")
            else:
                if dim != max_beats:
                    raise ValueError(f"Feature '{name}' has an unexpected beat count")
    
    # Trim every feature to the minimum beat count
    combined_feature = [f[:, :min_beats] for f in combined_feature]

    # Stack features vertically
    combined_feature = np.concatenate(combined_feature, axis=0)

    return combined_feature


if __name__ == '__main__':
    main()