DEBUG_MODE=False
NAME="extract_5_centerpad_nv0.4"
OUTPUT_DIR="./working"
DATA_ROOT="./Data"
FS=32000
WINDOW_SIZE=5
SEED=42
N_FFT=1024
HOP_LENGTH=128
N_MELS=128
FMIN=20
FMAX=16000
TARGET_SHAPE_X=512
TARGET_SHAPE_Y=512
BATCH_SIZE=16
USE_TTA=False
TTA_COUNT=3
TTA_THRES=5.0
USE_SMOOTHING=False
SMOOTHING_THRES=0.2
USE_SPECIFIC_FOLDS=False
FOLDS="0, 1, 2, 3, 4"
smooth_1=0.1
smooth_2=0.05

python3 infer.py \
  --debug_mode $DEBUG_MODE \
  --name $NAME \
  --output_dir $OUTPUT_DIR \
  --data_root $DATA_ROOT \
  --fs $FS \
  --window_size $WINDOW_SIZE \
  --seed $SEED \
  --n_fft $N_FFT \
  --hop_length $HOP_LENGTH \
  --n_mels $N_MELS \
  --fmin $FMIN \
  --fmax $FMAX \
  --target_shape $TARGET_SHAPE_X $TARGET_SHAPE_Y \
  --batch_size $BATCH_SIZE \
  --use_tta $USE_TTA \
  --tta_count $TTA_COUNT \
  --tta_thres $TTA_THRES \
  --use_smoothing $USE_SMOOTHING \
  --smoothing_thres $SMOOTHING_THRES \
  --use_specific_folds $USE_SPECIFIC_FOLDS \
  --folds "${FOLDS}" \
  --smooth_1 $smooth_1 \
  --smooth_2 $smooth_2


