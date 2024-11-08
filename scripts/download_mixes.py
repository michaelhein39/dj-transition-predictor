import pandas as pd
import os
import yt_dlp

def download_mixes():
    df = pd.read_csv('data/meta/tracklist_trunc.csv', skipinitialspace=True)
    mix_ids = df['mix_id'].unique()
    mix_df = pd.DataFrame({'mix_id': mix_ids})

    os.makedirs('data/mix/', exist_ok=True)

    for i, mix in mix_df.iterrows():
        mix_id = mix['mix_id']
        mix_url = df.loc[df['mix_id'] == mix_id, 'mix_audio_url'].iloc[0]


        # Define the output file path
        output_file = f'data/track/{i:02}_{e.mix_id}_{track.track_id}'

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

if __name__ == '__main__':
    download_mixes()