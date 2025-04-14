
DEBUG_MODE=False
NAME="11_10"
# 1,2,3 / 1,2,3 / 0,1 / 0,1
# only do 10/01
OUTPUT_DIR="./working/"
DATA_ROOT="./Data"
FS=32000
SEED=42
N_FFT=1024   # 1024 2048 4096
HOP_LENGTH=128 #FIXED
N_MELS=128   # 128, 256, 512
FMIN=20 # FIXED
FMAX=16000 # FIXED
EXCLUDE_HUMAN_VOICE=True
OVERSAMPLE_THRESHOLD=200
# not working
TARGET_DURATION=5.0 # FIXED
TARGET_SHAPE_X=512 # 256, 512
TARGET_SHAPE_Y=256 # 256, 512
PREPROCESSED_NV_DIR="./nvlist0.4.pkl"
# if provided, use pre-processed novoicelist. if not, process on-the-fly
PADDING="centerpad"
# Available options : "cyclic", "centerpad", "leftpad" for padding
EXTRACTION="random"
# Available options :  "random", "forward" for extraction
N_EXTRACT=1
# Available options : positive integers for n_extract, insert big number to get max samples
NORMALIZE=True

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
  --oversample_threshold $OVERSAMPLE_THRESHOLD \
  --target_duration $TARGET_DURATION \
  --target_shape $TARGET_SHAPE_X $TARGET_SHAPE_Y \
  --prepared_nv "${PREPROCESSED_NV_DIR}" \
  --padding $PADDING \
  --extraction $EXTRACTION \
  --n_extract $N_EXTRACT \
  --normalize $NORMALIZE