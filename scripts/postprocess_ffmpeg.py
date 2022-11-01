import os
import shutil

from tqdm import tqdm

os.makedirs('../output_mxn_eval_postprocessed_v1', exist_ok=True)
os.makedirs('../output_mxn_eval_postprocessed_v1/higtwg', exist_ok=True)

src_dir = '../output_mxn_eval/higtwg'
tgt_dir = '../output_mxn_eval_postprocessed_v1/higtwg/'
filenames = os.listdir(src_dir)

for filename in tqdm(filenames):
    src = os.path.join(src_dir, filename)
    tgt = os.path.join(tgt_dir ,filename.replace('.wav', '.wav'))
    os.system(f'ffmpeg -y -i {src} -af "highpass=85,lowpass=8000,afftdn" {tgt}')