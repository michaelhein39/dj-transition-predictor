import pandas as pd
import os
import yt_dlp

os.makedirs('data/mix/', exist_ok=True)
os.makedirs('data/track/', exist_ok=True)

df = pd.read_csv('data/meta/tracklist.csv', skipinitialspace=True)

for i, track in df.iterrows():
    if i == 2:
        break
    if track.audio_source != 'youtube':
        continue

    # Define the output file path
    output_file = f'data/track/{i:02}_{track.mix_id}_{track.track_id}'

    # Check if the file already exists
    if os.path.exists(output_file + '.wav'):
        print(f"File already exists: {output_file}")
        continue  # Skip downloading if file already exists

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