#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --mem 32G
#SBATCH --time 2-00:00:00
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
cd scripts_ssl
INPUT_TYPE="s2"
INPUT_DIR="/share/projects/siamdl/data/small/"
BATCH_SIZE=16
PROCESS_LEVEL="l1c"
PATIENCE=80
NUM_CLASSES=11
LR=0.0005
GAMMA=0.9
WEIGHT_DECAY=1e-7
EPOCHS=60
SSL_TYPE="dual"
OMEGA=0.5
GAMMA_SSL=0.9
LOSS_SSL_1="DiceLoss"
LOSS_SSL_2="L1Loss"
OUT_PATH="/share/projects/siamdl/outputs/${SLURM_JOBID}_$(date +%Y%m%d)_$SSL_TYPE/quickview/"
REMARKS="SSL Run."

# Create output directory
mkdir -p $OUT_PATH

# Save arguments to a file
echo "Arguments passed to the script:" > $OUT_PATH/arguments.txt
echo "Output Path: $OUT_PATH" >> $OUT_PATH/arguments.txt
echo "Input Type: $INPUT_TYPE" >> $OUT_PATH/arguments.txt
echo "Batch Size: $BATCH_SIZE" >> $OUT_PATH/arguments.txt
echo "Process Level: $PROCESS_LEVEL" >> $OUT_PATH/arguments.txt
echo "Omega: $OMEGA" >> $OUT_PATH/arguments.txt
echo "Patience: $PATIENCE" >> $OUT_PATH/arguments.txt
echo "Number of Classes: $NUM_CLASSES" >> $OUT_PATH/arguments.txt
echo "Learning Rate: $LR" >> $OUT_PATH/arguments.txt
echo "Weight Decay: $WEIGHT_DECAY" >> $OUT_PATH/arguments.txt
echo "Epochs: $EPOCHS" >> $OUT_PATH/arguments.txt
echo "Gamma: $GAMMA" >> $OUT_PATH/arguments.txt
echo "SSL Type: $SSL_TYPE" >> $OUT_PATH/arguments.txt
echo "Gamma SSL: $GAMMA_SSL" >> $OUT_PATH/arguments.txt
echo "SSL Loss 1: $LOSS_SSL_1" >> $OUT_PATH/arguments.txt
echo "SSL Loss 2: $LOSS_SSL_2" >> $OUT_PATH/arguments.txt
echo "Input Directory: $INPUT_DIR" >> $OUT_PATH/arguments.txt
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
    --input_type $INPUT_TYPE \
    --input_dir $INPUT_DIR \
    --batch_size $BATCH_SIZE \
    --process_level $PROCESS_LEVEL \
    --patience $PATIENCE \
    --num_classes $NUM_CLASSES \
    --lr $LR \
    --gamma $GAMMA \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --ssl_type $SSL_TYPE \
    --omega $OMEGA \
    --gamma_ssl $GAMMA_SSL \
    --loss_ssl_1 $LOSS_SSL_1 \
    --loss_ssl_2 $LOSS_SSL_2 \
    --out_path $OUT_PATH


# # Stop GPU monitoring process
# trap "kill $!" EXIT

cd
cp "siamdl${SLURM_JOBID}.out" $OUT_PATH
echo $(date)
