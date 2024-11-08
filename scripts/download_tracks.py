import pandas as pd
import os
import yt_dlp

def download_tracks():
    df = pd.read_csv('data/meta/tracklist.csv', skipinitialspace=True)

    os.makedirs('data/track/', exist_ok=True)

    for track in df.iterrows():
        mix_id = track['mix_id']
        track_id = track['track_id']
        i_track = track['i_track']
        audio_url = track['audio_url']

        # Define the output file path
        output_file = f'data/track/{mix_id}_{i_track:02}_{track_id}'

        # Skip downloading if file already exists
        if os.path.exists(f'{output_file}.wav'):
            print(f"File already exists: {output_file}")
            continue

        # yt-dlp options
        ydl_opts = {
            'outtmpl': output_file,
            'format': 'bestaudio/best',  # Get best available audio
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',  # Extract only audio
                'preferredcodec': 'wav',  # Save as .wav file
            }],
            'ratelimit': 3072 * 1024,  # Limit to 3 MB/s
            'sleep_requests': 1,  # Add a 2-second sleep between requests
            'sleep_interval': 2,  # Add a 2-second sleep between downloads
            'retry_sleep': {
                'fragment': 300  # Wait 5 minutes (300 seconds) on 429 HTTP error
            }
        }

        # Download the audio file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([audio_url])
                print(f'Downloaded: {output_file}.wav')
            except Exception as e:
                print(f'Error downloading {audio_url}: {e}')

if __name__ == "__main__":
    download_tracks()