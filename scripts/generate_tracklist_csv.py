import json
import pandas as pd

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
    track_info = track_dict.get(track_id)
    if track_info:
        artist_id = track_info['artist_id']
        title = track_info['title']
        bpm = track_info['bpm']
        key = track_info['key']
        audio_sr = track_info['audio_sr']
        audio_source = track_info['audio_source']
        audio_url = track_info['audio_url']

        # Find the corresponding artist data
        artist_info = artist_dict.get(artist_id)
        if artist_info:
            artist_name = artist_info['name']

            # Create a new row for the CSV file
            row = {
                'mix_id': mix_id,
                'i_track': f'{i_track:02}',
                'track_id': track_id,
                'timestamp': timestamp,
                'artist': artist_name,
                'title': title,
                'bpm': bpm,
                'key': key,
                'audio_sr': audio_sr,
                'audio_source': audio_source,
                'audio_url': audio_url,
                'filename': f'{i_track:02}_{mix_id}_{track_id}'
            }
            rows.append(row)
        else:
            print(f"artist_id {artist_id} from track_id {track_id} was not found in artist.json")
    else:
        print(f"track_id {track_id} from mix_id {mix_id} was not found in track.json")

# Create a Pandas DataFrame from the rows
df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
df.to_csv('data/meta/tracklist.csv', index=False)