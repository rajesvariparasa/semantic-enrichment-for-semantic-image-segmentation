#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --mem 32G
#SBATCH --time 30:00:00
#SBATCH --partition shortrun
#SBATCH --output=siamdl%j.out
#SBATCH --mail-type FAIL,END

if [[ ! -z ${SLURM_JOBID+z} ]]; then
    echo "Setting up SLURM environment"
    # Load the Conda environment
    source /share/common/anaconda/etc/profile.d/conda.sh
    conda activate torch_env
else
    echo "Not a SLURM job"
fi


set -o errexit -o pipefail -o nounset
cd scripts
INPUT_TYPE="siam_96"
INPUT_DIR="/share/projects/siamdl/data/small/"
OUT_PATH="/share/projects/siamdl/outputs/${SLURM_JOBID}_$(date +%Y%m%d)_$INPUT_TYPE/quickview/"
BATCH_SIZE=16
PROCESS_LEVEL="l1c"
LEARN_TYPE="csl"
PATIENCE=80
NUM_CLASSES=11
LR=0.0001
GAMMA=0.95
WEIGHT_DECAY=1e-7
EPOCHS=80
ENCODER_NAME="resnet50"
REMARKS="Increased epochs and gamma to 0.95. Encoder change."


# Create output directory
mkdir -p $OUT_PATH

# Save arguments to a file
echo "Arguments passed to the script:" > $OUT_PATH/arguments.txt
echo "Input Directory: $INPUT_DIR" >> $OUT_PATH/arguments.txt
echo "Output Path: $OUT_PATH" >> $OUT_PATH/arguments.txt
echo "Input Type: $INPUT_TYPE" >> $OUT_PATH/arguments.txt
echo "Batch Size: $BATCH_SIZE" >> $OUT_PATH/arguments.txt
echo "Process Level: $PROCESS_LEVEL" >> $OUT_PATH/arguments.txt
echo "Learn Type: $LEARN_TYPE" >> $OUT_PATH/arguments.txt
echo "Patience: $PATIENCE" >> $OUT_PATH/arguments.txt
echo "Number of Classes: $NUM_CLASSES" >> $OUT_PATH/arguments.txt
echo "Learning Rate: $LR" >> $OUT_PATH/arguments.txt
echo "Gamma: $GAMMA" >> $OUT_PATH/arguments.txt
echo "Weight Decay: $WEIGHT_DECAY" >> $OUT_PATH/arguments.txt
echo "Epochs: $EPOCHS" >> $OUT_PATH/arguments.txt
echo "Encoder Name: $ENCODER_NAME" >> $OUT_PATH/arguments.txt
echo "Remarks: $REMARKS" >> $OUT_PATH/arguments.txt

echo "Starting script"
echo $(date)

# monitor_gpu() {
#     output_file="$OUT_PATH/gpu_monitoring.csv"
    
#     # Check if the file exists, if not, add headers
#     if [ ! -f "$output_file" ]; then
#         echo "datetime,utilization_PC,temperature_C,power_draw_W" > "$output_file"
#     fi
    
#     while true; do
#         gpu_info=$(srun -s --jobid $SLURM_JOBID nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits)
#         gpu_utilization=$(echo $gpu_info | awk '{print $1}')
#         gpu_temperature=$(echo $gpu_info | awk '{print $2}')
#         power_draw=$(echo $gpu_info | awk '{print $3}')
#         echo "$(date +"%Y-%m-%d %H:%M:%S"),$gpu_utilization,$gpu_temperature,$power_draw" >> "$output_file"
#         sleep 10
#     done
# }


# # Start GPU monitoring in the background
# monitor_gpu &

python main.py \
    --input_dir $INPUT_DIR \
    --out_path $OUT_PATH \
    --input_type $INPUT_TYPE \
    --batch_size $BATCH_SIZE \
    --process_level $PROCESS_LEVEL \
    --learn_type $LEARN_TYPE \
    --patience $PATIENCE \
    --num_classes $NUM_CLASSES \
    --lr $LR \
    --gamma $GAMMA \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --encoder_name $ENCODER_NAME

# # Stop GPU monitoring process
# trap "kill $!" EXIT

cd
cp "siamdl${SLURM_JOBID}.out" $OUT_PATH
echo $(date)
