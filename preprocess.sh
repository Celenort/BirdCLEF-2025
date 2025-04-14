
DEBUG_MODE=True
NAME="Test1"
OUTPUT_DIR="./working/"
DATA_ROOT="./Data"
FS=32000
SEED=42
N_FFT=1024
HOP_LENGTH=512
N_MELS=128
FMIN=50
FMAX=14000
EXCLUDE_HUMAN_VOICE=True
NOHUMAN_DURATION=5.0
OVERSAMPLE_THRESHOLD=200
TARGET_DURATION=5.0
TARGET_SHAPE_X=256
TARGET_SHAPE_Y=256
PREPROCESSED_NV_DIR="./nvlist.pkl"
PADDING="centerpad"
EXTRACTION="random"
N_EXTRACT=1
NORMALIZE=True
# will support cyclic, zero padding for padding
# will support random, forward, center for extraction
# will support positive integers for n_extract, 0 for all

python3 preprocess.py \
  --debug_mode $DEBUG_MODE \
  --name $NAME \
  --output_dir $OUTPUT_DIR \
  --data_root $DATA_ROOT \
  --fs $FS \
  --seed $SEED \
  --n_fft $N_FFT \
  --hop_length $HOP_LENGTH \
  --n_mels $N_MELS \
  --fmin $FMIN \
  --fmax $FMAX \
  --exclude_human_voice $EXCLUDE_HUMAN_VOICE \
  --nohuman_duration $NOHUMAN_DURATION \
  --oversample_threshold $OVERSAMPLE_THRESHOLD \
  --target_duration $TARGET_DURATION \
  --target_shape $TARGET_SHAPE_X $TARGET_SHAPE_Y \
  --prepared_nv "${PREPROCESSED_NV_DIR}" \
  --padding $PADDING \
  --extraction $EXTRACTION \
  --n_extract $N_EXTRACT \
  --normalize $NORMALIZE