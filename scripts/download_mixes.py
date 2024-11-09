import pandas as pd
import os
import yt_dlp
from tqdm import tqdm

def download_mixes():
    df = pd.read_csv('data/meta/mixes_trunc.csv', skipinitialspace=True)

    os.makedirs('data/mix/', exist_ok=True)

    for _, mix in tqdm(df.iterrows(), total=len(df), desc='Downloading mixes'):
        mix_id = mix['mix_id']
        mix_url = mix['audio_url']

        # Define the output file path
        output_file = f'data/mix/{mix_id}'

        # Skip downloading if file already exists
        if os.path.exists(f'{output_file}.wav'):
            print(f"File already exists: {output_file}.wav")
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
            'sleep_requests': 1,  # Add a 1-second sleep between requests
            'sleep_interval': 2,  # Add a 2-second sleep between downloads
            'retry_sleep': {
                'fragment': 300  # Wait 5 minutes (300 seconds) on 429 HTTP error
            }
        }

        # Download the audio file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([mix_url])
            except Exception as e:
                print(f'Error downloading {mix_url}: {e}')

if __name__ == "__main__":
    download_mixes()