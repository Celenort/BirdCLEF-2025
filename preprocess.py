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
import argparse
from dataclasses import dataclass
warnings.filterwarnings("ignore")


@dataclass
class Config:
    DEBUG_MODE: bool
    NAME: str
    OUTPUT_DIR: str
    DATA_ROOT: str
    FS: int
    SEED: int
    N_FFT: int
    HOP_LENGTH: int
    N_MELS: int
    FMIN: int
    FMAX: int
    EXCLUDE_HUMAN_VOICE: bool
    NOHUMAN_DURATION: float
    OVERSAMPLE_THRESHOLD: int
    TARGET_DURATION: float
    TARGET_SHAPE: tuple
    PREPARED_NV: str
    PADDING: str
    EXTRACTION: str

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug_mode', type=str2bool, default=True)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--fs', type=int, default=32000)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)

    parser.add_argument('--exclude_human_voice', type=str2bool, default=True)
    parser.add_argument('--nohuman_duration', type=float, default=5.0)

    parser.add_argument('--oversample_threshold', type=int, default=200)
    parser.add_argument('--target_duration', type=float, default=5.0)
    parser.add_argument('--target_shape', type=int, nargs=2, default=[256, 256])

    parser.add_argument('--prepared_nv', type=str, default="./nvlist.pkl")
    parser.add_argument('--padding', type=str, choices=["zero", "cyclic"], default="zero")
    parser.add_argument('--extraction', type=str, choices=["random", "forward", "center"], default="random")

    return parser.parse_args()

def get_config():
    args = parse_args()
    config = Config(
        DEBUG_MODE=args.debug_mode,
        NAME=args.name,
        OUTPUT_DIR=args.output_dir,
        DATA_ROOT=args.data_root,
        FS=args.fs,
        SEED=args.seed,
        N_FFT=args.n_fft,
        HOP_LENGTH=args.hop_length,
        N_MELS=args.n_mels,
        FMIN=args.fmin,
        FMAX=args.fmax,
        EXCLUDE_HUMAN_VOICE=args.exclude_human_voice,
        NOHUMAN_DURATION=args.nohuman_duration,
        OVERSAMPLE_THRESHOLD=args.oversample_threshold,
        TARGET_DURATION=args.target_duration,
        TARGET_SHAPE=tuple(args.target_shape),
        PREPARED_NV=args.prepared_nv,
        PADDING=args.padding,
        EXTRACTION=args.extraction
    )
    return config

def voice2novoice (config, vdict) :
    nv_file_dict = {}
    print(f'Found {len(vdict)} files')
    for (dir, vlist) in tqdm(vdict.items(), total=len(vdict)) :#vdict.items() :
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
    return nv_file_dict

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


if __name__ == '__main__':
    config = get_config()
    random.seed(config.SEED)

    print("Loading data...")
    taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
    species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
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
    total_samples = 50 if config.DEBUG_MODE else len(working_df)
    print(f'Total samples to process: {total_samples} out of {len(working_df)} available')

    print("Loading human voice data...")
    nv_file_dict = None
    if config.PREPARED_NV == "" :
        print("Preprocessed NoVoice pickle NOT detected")
        with open("train_voice_data.pkl", "rb") as fr :
            voice_dict = pickle.load(fr)
            voice_file_dict = {config.DATA_ROOT+key[27:]: value for key, value in voice_dict.items()} # remove /kaggle/input/birdclef-2025/train_audio/
            nv_file_dict = voice2novoice(config, voice_file_dict)
            with open("nvlist.pkl", "wb") as f :
                pickle.dump(nv_file_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        with open(config.PREPARED_NV, "rb") as fr :
            nv_file_dict = pickle.load(fr)
    print("Finished loading human voice data.")





    ###############################################################################

    
    print("Starting audio processing...")
    print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
    start_time = time.time()

    all_bird_data = {}
    errors = []
    skipcount  = 0

    for i, row in tqdm(working_df.iterrows(), total=total_samples): # working_df.iterrows(): 
        if config.DEBUG_MODE and i >= 50 : # previously config.N_MAX but hard setting this value does not change anything
            break
        
        try:
            audio_data, _ = librosa.load(row.filepath, sr=config.FS)

            target_samples = int(config.TARGET_DURATION * config.FS)

            start_frame = -1
            #voice_file_dict = {config.DATA_ROOT+key[:]: value for key, value in nv_file_dict.items()}

            if row.filename in nv_file_dict :
                nvlist = nv_file_dict[row.filename]
                if len(nvlist) == 0 :
                    # too short, also contains human voice
                    skipcount += 1
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
                if config.EXTRACTION == "random" :
                    start_idx = random.randint(0, total-1)
                elif config.EXTRACTION == "first" :
                    start_idx = 0
                else :
                    raise Exception('Only supports \"random\"and \"first\"')
                for start, length in lengths :
                    if start_idx < length :
                        start_frame = start + start_idx
                        break
                    start_idx -= length

            if len(audio_data) < target_samples:
                if config.PADDING == "centerpad" :
                    left_pad = int((target_samples - len(audio_data))/2)
                    right_pad = int(target_samples-len(audio_data)-left_pad)
                    audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant') # zero padding at center
                elif config.PADDING == "leftpad" :
                    audio_data = np.pad(audio_data, (int(target_samples-len(audio_data)), 0), mode='constant')
                elif config.PADDING == "cyclic" :
                    n_copy = math.ceil(target_samples / len(audio_data))
                    audio_data = np.concatenate([audio_data]* n_copy)
                else :
                    raise Exception('Only supports \"centerpad\", \"leftpad\" and \"cyclic\"')
            if start_frame == -1 : # didn't pass through nv
                if config.EXTRACTION == "random" :
                    start_frame = random.randint(0, len(audio_data)-target_samples)
                elif config.EXTRACTION == "first" :
                    start_frame = 0
                else :
                    raise Exception('Only supports \"random\"and \"first\"')
            
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

    np.save(config.NAME + '.npy', all_bird_data)




################################################################################


## Changelog : Remove cyclic padding, only add zero padding (equally on both sides)
