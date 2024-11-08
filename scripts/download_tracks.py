import pandas as pd
import os
import yt_dlp

os.makedirs('data/mix/', exist_ok=True)
os.makedirs('data/track/', exist_ok=True)

df = pd.read_csv('data/meta/tracklist.csv', skipinitialspace=True)

for i, track in df.iterrows():
    if i == 2:
        break

    # Define the output file path
    output_file = f'data/track/{i:02}_{track.mix_id}_{track.track_id}'

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
        'ratelimit': 50 * 1024,  # Limit to 50 KB/s
    }

    # Download the audio file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([track.audio_url])

    # Download the audio file from the given URL
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([track.audio_url])
            print(f'Downloaded: {output_file}.wav')
        except Exception as e:
            print(f'Error downloading {track.audio_url}: {e}')