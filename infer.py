import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from dataclasses import dataclass
from tqdm.auto import tqdm
import argparse

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

###################################################

@dataclass
class CFG:
    debug: bool
    NAME: str
    OUTPUT_DIR: str
    DATA_ROOT: str
    FS: int
    WINDOW_SIZE: int
    SEED: int
    N_FFT: int
    HOP_LENGTH: int
    N_MELS: int
    FMIN: int
    FMAX: int
    TARGET_SHAPE: tuple
    batch_size: int
    use_tta: bool
    tta_count: int
    tta_thres: int
    use_smoothing: bool
    smoothing_thres: float
    use_specific_folds: bool
    folds: list
    model_name: str
    in_channels: int
    device: str
    debug_count: int
    test_soundscapes: str
    submission_csv: str
    taxonomy_csv: str
    model_path: str
    smooth_1: float
    smooth_2: float

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
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--hop_length', type=int, default=512)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)

    parser.add_argument('--target_shape', type=int, nargs=2, default=[256, 256])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_tta', type=str2bool, default=False)
    parser.add_argument('--tta_count', type=int, default=3)
    parser.add_argument('--tta_thres', type=float, default=0.5)
    parser.add_argument('--use_smoothing', type=str2bool, default=False)
    parser.add_argument('--smoothing_thres', type=float, default=0.2)
    parser.add_argument('--use_specific_folds', type=str2bool, default=False)
    parser.add_argument('--folds', type=str, default="0,")
    parser.add_argument('--smooth_1', type=float, default=0.2)
    parser.add_argument('--smooth_2', type=float, default=None)

    return parser.parse_args()

def get_config():
    args = parse_args()
    config = CFG(
        debug=args.debug_mode,
        NAME=args.name,
        OUTPUT_DIR=args.output_dir,
        DATA_ROOT=args.data_root,
        FS=args.fs,
        WINDOW_SIZE=args.window_size,
        SEED=args.seed,
        N_FFT=args.n_fft,
        HOP_LENGTH=args.hop_length,
        N_MELS=args.n_mels,
        FMIN=args.fmin,
        FMAX=args.fmax,
        TARGET_SHAPE=tuple(args.target_shape),
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        tta_count=args.tta_count,
        tta_thres=args.tta_thres,
        use_smoothing=args.use_smoothing,
        smoothing_thres=args.smoothing_thres,
        use_specific_folds=args.use_specific_folds,
        folds=args.folds.split(","), 
        model_name='efficientnet_b0',
        in_channels=1,
        device='cpu',
        debug_count=5, 
        test_soundscapes='/train_soundscapes',
        submission_csv='/sample_submission.csv',
        taxonomy_csv = '/taxonomy.csv',
        model_path = './weight-1024-128-512-512', 
        smooth_1 = args.smooth_1,
        smooth_2 = args.smooth_2
    )
    return config
###################################################
# test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
# submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
# taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
# model_path = '/kaggle/input/weight-1024-128-256-5124'
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        if isinstance(features, dict):
            features = features['features']
            
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)
        
        logits = self.classifier(features)
        return logits

def smooth_submission(submission_path, coeff1, coeff2=None):
    """
    Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
    
    For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
    are averaged with those of its neighbors using defined weights.
    
    :param submission_path: Path to the submission CSV file.
    """
    print("Smoothing submission predictions...")
    sub = pd.read_csv(submission_path)
    cols = sub.columns[1:]
    # Extract group names by splitting row_id on the last underscore
    groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
    unique_groups = np.unique(groups)
        
    for group in unique_groups:
        # Get indices for the current group
        idx = np.where(groups == group)[0]
        sub_group = sub.iloc[idx].copy()
        predictions = sub_group[cols].values
        new_predictions = predictions.copy()
        if not coeff2 and predictions.shape[0] > 1:
            # Smooth the predictions using neighboring segments
            new_predictions[0] = (predictions[0] * (1.0-coeff1)) + (predictions[1] * coeff1)
            new_predictions[-1] = (predictions[-1] * (1.0-coeff1)) + (predictions[-2] * coeff1)
            for i in range(1, predictions.shape[0]-1):
                new_predictions[i] = (predictions[i-1] * (coeff1)) + (predictions[i] * (1.0- 2 * coeff1)) + (predictions[i+1] * coeff1)    
        elif predictions.shape[0] > 3:
            # Smooth the predictions using neighboring segments
            new_predictions[0] = (predictions[0] * (1.0-coeff1-coeff2)) + (predictions[1] * coeff1) + (predictions[2] * coeff2)
            new_predictions[1] = (predictions[0] * coeff1) + (predictions[1] * (1.0-2 * coeff1-coeff2)) + (predictions[2] * coeff1) + (predictions[3] * coeff2)
            new_predictions[-1] = (predictions[-1] * (1.0-coeff1-coeff2)) + (predictions[-2] * coeff1) + (predictions[-3] * coeff2)
            new_predictions[-2] = (predictions[-1] * coeff1) + (predictions[-2] * (1.0-2 * coeff1-coeff2)) + (predictions[-3] * coeff1) + (predictions[-4] * coeff2)
            for i in range(2, predictions.shape[0]-2):
                new_predictions[i] = (predictions[i-2] * coeff2) + (predictions[i-1] * coeff1) + (predictions[i] * (1.0-2 * (coeff1+coeff2))) + (predictions[i+1] * coeff1) + (predictions[i+2] * coeff2)
        # Replace the smoothed values in the submission dataframe
        sub.iloc[idx, 1:] = new_predictions
        
    sub.to_csv(submission_path, index=False)
    print(f"Smoothed submission saved to {submission_path}")

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')
    
    mel_spec = audio2melspec(audio_data, cfg)
    
    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)

def find_model_files(cfg):
    """
    Find all .pth model files in the specified model directory
    """
    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    model_files = find_model_files(cfg)
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        # print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            if cfg.use_tta:
                all_preds = []
                
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)
                        
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                final_preds = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, cfg)
                
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)
                
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)
                    
            predictions.append(final_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions

def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0)
    else:
        return spec

def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    print(cfg.DATA_ROOT + cfg.test_soundscapes)
    test_files = list(Path(cfg.DATA_ROOT + cfg.test_soundscapes).glob('*.ogg'))
    
    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.DATA_ROOT+cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df

def main():

    cfg = get_config()


    print(f"Using device: {cfg.device}")
    print(f"Loading taxonomy data...")
    taxonomy_df = pd.read_csv(cfg.DATA_ROOT+cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    num_classes = len(species_ids)
    print(f"Number of classes: {num_classes}")

    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

    models = load_models(cfg, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    if cfg.use_smoothing :
        smooth_submission(submission_path, cfg.smooth_1, cfg.smooth_2)

    print(f"Submission saved to {submission_path}")
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()
