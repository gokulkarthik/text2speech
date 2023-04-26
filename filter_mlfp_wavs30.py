"""Coverage: ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'ta', 'te']"""
import os
import shutil

from tqdm import tqdm

dest_dir = 'output_mlfp_30_wavs'
os.makedirs(dest_dir, exist_ok=True)

with open('output_mlfp_wavs/eval30list.txt', 'r') as f:
    files = f.read().splitlines()

def extract_language(s):
    language_speaker = s.split('_')[1]
    splitter = 'female' if 'female' in s else 'male'
    language = language_speaker.replace(splitter, '').replace('telugu13x', 'telugu').replace('full','')
    return language

language2iso = {
    'Assamese': 'as',
    'Bengali': 'bn',
    'Bodo': 'brx',
    'Gujarati': 'gu',
    'Hindi': 'hi',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Manipuri': 'mni',
    'Marathi': 'mr',
    'Odia': 'or',
    'Rajasthani': 'raj',
    'Tamil': 'ta',
    'Telugu': 'te',
}

uniq_langs = []

for file in tqdm(files):
    language = extract_language(file)
    iso = language2iso[language.title()]

    if iso in ['kn', 'ml', 'ta', 'te']:
        src_dir = f'output_mlfp_wavs/samples_mlfpdravidian_indictts_{iso}'
    else:
        src_dir = f'output_mlfp_wavs/samples_mlfpindoaryan_indictts_{iso}'
    
    if not os.path.exists(src_dir):
        continue
    
    src = os.path.join(src_dir, file)
    if not os.path.exists(src):
        continue
    
    uniq_langs.append(iso)
    dest = os.path.join(dest_dir, file)
    shutil.copy2(src, dest)

print(sorted(set(uniq_langs)))