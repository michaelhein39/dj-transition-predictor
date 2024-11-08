import json
import pandas as pd

def generate_mixes_csv():
    # Load the JSON files
    with open('data/meta/tracklist.json', 'r') as f:
        tracklist_data = json.load(f)
    with open('data/meta/mix.json', 'r') as f:
        mix_data = json.load(f)

    # Convert mix_data to a dictionary for faster lookup
    mix_dict = {m['mix_id']: m for m in mix_data}

    # Get unique mix_ids from tracklist_data
    unique_mix_ids = list(set([item['mix_id'] for item in tracklist_data]))

    # Create an empty list to store the rows of the new CSV file
    rows = []

    # Loop through each unique mix_id
    for mix_id in unique_mix_ids:
        # Find the corresponding mix data
        mix_info = mix_dict[mix_id]

        # Create a new row for the CSV file
        row = {
            'mix_id': mix_id,
            # 'meta_url': mix_info['meta_url'],
            'audio_url': mix_info['audio_url'],
            'audio_length': mix_info['audio_length'],
            'audio_size': mix_info['audio_size'],
            'audio_sr': mix_info['audio_sr'],
            'audio_source': mix_info['audio_source'],
            # 'title': mix_info['title'],
            # 'music_style': mix_info['music_style'],
            # 'num_views': mix_info['num_views'],
            # 'num_likes': mix_info['num_likes'],
            'num_tracks': mix_info['num_tracks'],
            'all_ided': mix_info['all_ided'],
            'num_ided_tracks': mix_info['num_ided_tracks'],
            # 'played_date': mix_info['played_date'],
            # 'posted_time': mix_info['posted_time'],
            # 'last_updated_time': mix_info['last_updated_time'],
            'dj_id': mix_info['dj_id']
        }
        rows.append(row)

    # Create a Pandas DataFrame from the rows
    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv('data/meta/mixes.csv', index=False)

if __name__ == '__main__':
    generate_mixes_csv()