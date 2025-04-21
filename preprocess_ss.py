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
#import torch
import warnings
import argparse
from dataclasses import dataclass
import glob
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
    OVERSAMPLE_THRESHOLD: int
    TARGET_DURATION: float
    TARGET_SHAPE: tuple
    PREPARED_NV: str
    VOICE_DIR: str
    NV_OUT_DIR: str
    PADDING: str
    EXTRACTION: str
    N_EXTRACT: int
    NORMALIZE: str

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
    parser.add_argument('--oversample_threshold', type=int, default=200)
    parser.add_argument('--target_duration', type=float, default=5.0)
    parser.add_argument('--target_shape', type=int, nargs=2, default=[256, 256])

    parser.add_argument('--prepared_nv', type=str, default="./nvlist.pkl")
    parser.add_argument('--padding', type=str, choices=["leftpad", "centerpad", "cyclic"], default="centerpad")
    parser.add_argument('--extraction', type=str, choices=["random", "forward"], default="random")

    parser.add_argument('--n_extract', type=int, default=1) # big number to extract all
    parser.add_argument('--normalize', type=str2bool, default=True)


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
        OVERSAMPLE_THRESHOLD=args.oversample_threshold,
        TARGET_DURATION=args.target_duration,
        TARGET_SHAPE=tuple(args.target_shape),
        PREPARED_NV=args.prepared_nv,
        VOICE_DIR = './train_voice_data_0.4.pkl',
        NV_OUT_DIR = './nvlist_0.4.pkl',
        PADDING=args.padding,
        EXTRACTION=args.extraction,
        N_EXTRACT=args.n_extract,
        NORMALIZE=args.normalize
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

def audio2melspec(audio_data, donorm: True):
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
    if donorm :
        return (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    else : 
        return mel_spec_db

def sample_from_ranges(ranges, length, n_segment, israndom=True):
    segment_infos = []
    total_slots = 0

    for r in ranges:
        start = r['start']
        end = r['end']
        rlength = end - start + 1
        max_in_segment = rlength // length
        if max_in_segment > 0:
            segment_infos.append({
                'start': start,
                'length': length,
                'max_chunks': max_in_segment,
                'global_start_index': total_slots  # offset for mapping later
            })
            total_slots += max_in_segment

    if total_slots == 0:
        return None

    n_segment = min(n_segment, total_slots)

    if not israndom:
        intervals = []
        remaining = n_segment
        for seg in segment_infos:
            usable = min(seg['max_chunks'], remaining)
            for i in range(usable):
                s = seg['start'] + i * length
                intervals.append((s, s + length - 1))
            remaining -= usable
            if remaining == 0:
                break
        return intervals

    buffer = total_slots - n_segment

    cuts = sorted(random.sample(range(buffer + n_segment), n_segment))
    gaps = [cuts[0]] + [cuts[i] - cuts[i - 1] - 1 for i in range(1, n_segment)] + [buffer + n_segment - 1 - cuts[-1]]

    global_positions = []
    current = gaps[0]
    for i in range(n_segment):
        global_positions.append(current)
        current += 1 + gaps[i + 1]

    intervals = []
    for gpos in global_positions:
        for seg in segment_infos:
            if gpos < seg['global_start_index'] + seg['max_chunks']:
                local_index = gpos - seg['global_start_index']
                actual_start = seg['start'] + local_index * length
                intervals.append((actual_start, actual_start + length - 1))
                break

    return intervals

def pad_short_clips(audio_data, target_samples, padding) :
    if padding == "centerpad" :
        left_pad = int((target_samples - len(audio_data))/2)
        right_pad = int(target_samples-len(audio_data)-left_pad)
        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant') # zero padding at center
    elif padding == "leftpad" :
        audio_data = np.pad(audio_data, (int(target_samples-len(audio_data)), 0), mode='constant')
    elif padding == "cyclic" :
        n_copy = math.ceil(target_samples / len(audio_data))
        audio_data = np.concatenate([audio_data]* n_copy)
    return audio_data


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

    filepaths = glob.glob(config.DATA_ROOT + '/train_soundscapes/**.ogg')
    total_samples = 50 if config.DEBUG_MODE else len(filepaths)
    print(f'Total samples to process: {total_samples} available')


    print("Starting audio processing...")
    print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
    start_time = time.time()

    # all_bird_data = {}
    errors = []
    padcount = 0
    cols=['filedir', 'filename', 'index', 'startidx']
    working_csv = pd.DataFrame(columns=cols)

    errorlog = open(config.NAME+'_err.txt', "w")

    for i, row in tqdm(enumerate(filepaths), total=total_samples): # working_df.iterrows(): 
        if config.DEBUG_MODE and i >= 50 : # previously config.N_MAX but hard setting this value does not change anything
            break
        
        try:
            audio_data, _ = librosa.load(row, sr=config.FS)

            target_samples = int(config.TARGET_DURATION * config.FS)

            if len(audio_data) < target_samples:
                audio_data = pad_short_clips(audio_data, target_samples, config.PADDING)
                padcount+=1

            available_range = []

            if config.EXCLUDE_HUMAN_VOICE :
                available_range = nv_file_dict[row.filename] # cant go
            else :
                available_range = [{'start': 0, 'end': len(audio_data)-1}]
            indexes = sample_from_ranges(available_range, target_samples, config.N_EXTRACT, config.EXTRACTION)

            if indexes is None:
                raise ValueError("Not enough spaces to cut")


            for j, index in enumerate(indexes) :

                cropped_audio = audio_data[index[0]:index[1]+1]
                mel_spec = audio2melspec(cropped_audio, config.NORMALIZE)
                if mel_spec.shape != config.TARGET_SHAPE:
                    mel_spec = cv2.resize(mel_spec, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
                # sample_label = row.samplename + '-' + str(j)
                # all_bird_data[sample_label] = mel_spec.astype(np.float32)

                fullpath = config.OUTPUT_DIR + '/' + config.NAME + '/' + row[25:] + '-' +str(j) + '.npy'

                directory = '/'.join(fullpath.split('/')[:-1])

                if not os.path.exists(directory):
                    os.makedirs(directory)
        
                np.save(fullpath, mel_spec.astype(np.float32))


                newrow = pd.DataFrame([[None, None, None, None]], columns=cols)
                fdir = row
                newrow['startidx'] = index[0]
                newrow['filename'] = fdir[25:]
                newrow['filedir'] = fdir
                newrow['index'] = j
                #print(newrow['samplename'])
                working_csv = pd.concat([working_csv, newrow], ignore_index=True)

            
        except Exception as e:
            errorlog.write(f"Error processing {row}: {e}\n")
            errors.append((row, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    # print(f"Successfully processed {len(all_bird_data)} clips out of {total_samples} total")
    print(f"Padded {padcount} files because it was too short")
    print(f"Failed to process {len(errors)} files")

    # np.save(config.NAME + '.npy', all_bird_data)

    with open(config.NAME+'.txt', "w") as file:
        file.write(f"Processing completed in {end_time - start_time:.2f} seconds\n")
        file.write(f"Padded {padcount} files because it was too short\n")
        file.write(f"Failed to process {len(errors)} files\n")

    working_csv.to_csv(config.NAME + '.csv', index=False)
    errorlog.close()
