import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import random
import torch
import warnings
warnings.filterwarnings("ignore")

class Config:
 
    DEBUG_MODE = False
    
    OUTPUT_DIR = './working/'
    DATA_ROOT = './Data'
    FS = 32000
    
    SEED = 42

    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000

    EXCLUDE_HUMAN_VOICE = True
    NOHUMAN_DURATION = 5.0

    OVERSAMPLE_THRESHOLD = 200
    
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)  
    
    N_MAX = 50 if DEBUG_MODE else None  

config = Config()
random.seed(config.SEED)

print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
print(f"Max samples to process: {config.N_MAX if config.N_MAX is not None else 'ALL'}")

print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

print("Load vocal data...")
with open("train_voice_data.pkl", "rb") as fr :
    voice_dict = pickle.load(fr)

print("Loading training metadata...")
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')

label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')
working_df = train_df[['primary_label', 'rating', 'filename']].copy()
working_df['target'] = working_df.primary_label.map(label2id)
working_df['filepath'] = config.DATA_ROOT + '/train_audio/' + working_df.filename
working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
total_samples = min(len(working_df), config.N_MAX or len(working_df))
print(f'Total samples to process: {total_samples} out of {len(working_df)} available')
print(f'Samples by class:')
print(working_df['class'].value_counts())

# voice_file_dict = {key[40:]: value for key, value in voice_dict.items()} # remove /kaggle/input/birdclef-2025/train_audio/

voice_file_dict = {config.DATA_ROOT+key[27:]: value for key, value in voice_dict.items()} # remove /kaggle/input/birdclef-2025/train_audio/

#audio_data, _ = librosa.load(row.filepath, sr=config.FS)
nv_file_dict = {}
print(f'Found {len(voice_file_dict)} files')
for (dir, vlist) in tqdm(voice_file_dict.items(), total=len(voice_file_dict)) :#voice_file_dict.items() :
    audio_file, _ = librosa.load(dir, sr=config.FS)
    lenaudio = len(audio_file)
    nvlist = []
    nvlist.append({'start' : 0, 'end' : vlist[0]['start']-1})
    for i in range(len(vlist)-1) :
        nvlist.append({'start' : vlist[i]['end']+1, 'end' : vlist[i+1]['start']-1})
    if len(vlist)==1 :
        nvlist.append({'start' : vlist[0]['end']+1, 'end' : lenaudio})
    else :
        nvlist.append({'start' : vlist[i+1]['end']+1, 'end' : lenaudio})
    # check for too short count
    
    for j in reversed(range(len(nvlist))) :
        start = nvlist[j]['start']
        end = nvlist[j]['end']
        if (start+config.FS * config.TARGET_DURATION) >= end :
            nvlist.pop(j)
            # remove too short ones
    nv_file_dict[dir[19:]] = nvlist

# nv_file_dict : same format but stores no-voice range
print("Finished creating no-voice list")

def audio2melspec(audio_data):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    mel_spec_power_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    
    return (mel_spec_norm, mel_spec_db, mel_spec_norm, mel_spec_power_norm)

## Changelog : Remove cyclic padding, only add zero padding (equally on both sides)

print("Starting audio processing...")
print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
start_time = time.time()

all_bird_data = {}
errors = []
skipcount  = 0

for i, row in tqdm(working_df.iterrows(), total=total_samples): # working_df.iterrows(): 
    if config.N_MAX is not None and i >= config.N_MAX:
        break
    
    try:
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)

        target_samples = int(config.TARGET_DURATION * config.FS)

        start_frame = -1

        if row.filename in voice_file_dict :
            nvlist = nv_file_dict[row.filename]
            if len(nvlist) == 0 :
                # too short, also contains human voice
                skipcount = skipcount+1
                continue
            for i in range(len(nvlist)):
                nvlist[i]['end'] = nvlist[i]['end']-target_samples
                # narrowing range for start idx
            
            # let's choose among nvlist!
            lengths = []
            total = 0
            for r in nvlist:
                length = r['end']-r['start']+1
                lengths.append((r['start'], length))
                total += length
            start_idx = random.randint(0, total-1)
            for start, length in lengths :
                if start_idx < length :
                    start_frame = start + start_idx
                    break
                start_idx -= length

        if len(audio_data) < target_samples:
            left_pad = int((target_samples - len(audio_data))/2)
            right_pad = int(target_samples-len(audio_data)-left_pad)
            audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        if start_frame == -1 : # didn't pass through nv
            start_frame = random.randint(0, len(audio_data)-target_samples)
        
        cropped_audio = audio_data[start_frame:start_frame+target_samples]

        mel_spec = audio2melspec(cropped_audio)[0]

        if mel_spec.shape != config.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        all_bird_data[row.samplename] = mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {row.filepath}: {e}")
        errors.append((row.filepath, str(e)))

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {len(all_bird_data)} files out of {total_samples} total")
print(f'Skipped processing too short {skipcount} files')
print(f"Failed to process {len(errors)} files")

np.save('nv_strict_random.npy', all_bird_data)