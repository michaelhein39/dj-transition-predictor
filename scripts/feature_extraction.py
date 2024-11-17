import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
from lib.feature import *

# df_tlist = pd.read_csv('data/meta/tracks_trunc.csv')


def main():
  with Pool(os.cpu_count() // 2) as pool:
    # paths = [f'data/mix/{mix_id}.wav' for mix_id in df_tlist.mix_id.unique()]
    # paths += [f'data/track/{filename}.wav'  for filename in df_tlist.filename]
    mix_paths = glob.glob('data/mix/*.wav')
    track_paths = glob.glob('data/track/*.wav')
    paths = mix_paths + track_paths
    iterator = pool.imap(extract_feature, paths)
    for _ in tqdm(iterator, total=len(paths)):
      pass


def extract_feature(path):
  beat_chroma_cens(path)
  beat_mfcc(path)


if __name__ == '__main__':
  main()
