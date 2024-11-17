import json
import pandas as pd

def generate_tracks_csv():
    # Load the JSON files
    with open('data/meta/tracklist.json', 'r') as f:
        tracklist_data = json.load(f)
    with open('data/meta/track.json', 'r') as f:
        track_data = json.load(f)
    with open('data/meta/artist.json', 'r') as f:
        artist_data = json.load(f)

    # Convert track_data and artist_data to dictionaries for faster lookup
    track_dict = {t['track_id']: t for t in track_data}
    artist_dict = {a['artist_id']: a for a in artist_data}

    # Create an empty list to store the rows of the new CSV file
    rows = []

    # Loop through each item in the tracklist data
    for item in tracklist_data:
        mix_id = item['mix_id']
        i_track = item['i_track']
        track_id = item['track_id']
        timestamp = item['timestamp']

        # Find the corresponding track data
        track_info = track_dict[track_id]

        # Find the corresponding artist data
        artist_info = artist_dict.get(track_info['artist_id'])
        artist_name = artist_info['name'] if artist_info else None

        # Create a new row for the CSV file
        row = {
            'mix_id': mix_id,
            'i_track': f'{i_track:02}',
            'track_id': track_id,
            'timestamp': timestamp,
            'artist': artist_name,
            'title': track_info['title'],
            'bpm': track_info['bpm'],
            'key': track_info['key'],
            'audio_sr': track_info['audio_sr'],
            'audio_source': track_info['audio_source'],
            'audio_url': track_info['audio_url'],
            # 'meta_url': track_info['meta_url'],
            'audio_length': track_info['audio_length'],
            'audio_size': track_info['audio_size'],
            # 'music_style': track_info['music_style'],
            # 'num_views': track_info['num_views'],
            # 'num_likes': track_info['num_likes'],
            # 'played_date': track_info['played_date'],
            # 'posted_time': track_info['posted_time'],
            # 'last_updated_time': track_info['last_updated_time'],
            'filename': f'{mix_id}_{i_track:02}_{track_id}'
        }
        rows.append(row)

    # Create a Pandas DataFrame from the rows
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv('data/meta/tracks.csv', index=False)

if __name__ == '__main__':
    generate_tracks_csv()