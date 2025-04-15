
DEBUG_MODE=False
# NAME="extract_5_centerpad_nv0.4"
OUTPUT_DIR="./working"
DATA_ROOT="./Data"
FS=32000
SEED=42
# N_FFT=1024
HOP_LENGTH=128
# N_MELS=128
FMIN=20
FMAX=16000
EXCLUDE_HUMAN_VOICE=True
OVERSAMPLE_THRESHOLD=200
# not implemented
TARGET_DURATION=5.0
# TARGET_SHAPE_X=256
# TARGET_SHAPE_Y=256
PREPROCESSED_NV_DIR="./nvlist0.4.pkl"
# if provided, use pre-processed novoicelist. if not, process on-the-fly
PADDING="centerpad"
# Available options : "cyclic", "centerpad", "leftpad" for padding
EXTRACTION="random"
# Available options :  "random", "forward" for extraction
N_EXTRACT=1
# Available options : positive integers for n_extract, insert big number to get max samples
NORMALIZE=True

LOGFILE="execution_log.txt"


for N_FFT in 2048 4096; do # 2048 4096; do
    for N_MELS in 128 256 512; do #128 256 
        TARGET_SHAPE_X=256
        TARGET_SHAPE_Y=512
        
        START_TIME=$(date +%s)
        START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
                        
        NAME="${N_FFT}_${N_MELS}_${TARGET_SHAPE_X}_${TARGET_SHAPE_Y}"
        echo "Start processing : $NAME ($START_HUMAN)"


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


        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "$NAME | Start: $START_HUMAN | End: ${DURATION}s" >> "$LOGFILE"


        cd $OUTPUT_DIR/$NAME
        zip -r ../$NAME.zip ./
        cd ..
        rm -rf $NAME
        cd ..

    done
done



