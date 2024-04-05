#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --constraint m48
#SBATCH --mem 10G
#SBATCH --time 10:00:00
#SBATCH --partition shortrun
#SBATCH --output=siamdl_seg%j.out
#SBATCH --mail-type FAIL,END

if [[ ! -z ${SLURM_JOBID+z} ]]; then
    echo "Setting up SLURM environment"
    # Load the Conda environment
    source /share/common/anaconda/etc/profile.d/conda.sh
    conda activate pytorch-env
else
    echo "Not a SLURM job"
fi

set -o errexit -o pipefail -o nounset

INPUT_DIR="/share/projects/siamdl/data/small/"
OUT_PATH="/share/projects/siamdl/TestRun1/outputs"
BATCH_SIZE=16
PROCESS_LEVEL= "l1c"
LEARN_TYPE = "csl"
PATIENCE =5
NUM_CHANNELS = 10
NUM_CLASSES = 10
LR=0.0001
WEIGHT_DECAY=1e-7
EPOCHS=5

echo "Starting script"
echo $(date)

python main.py \
    --input_dir $INPUT_DIR \
    --out_path $OUT_PATH \
    --batch_size $BATCH_SIZE \
    --process_level $PROCESS_LEVEL \
    --learn_type $LEARN_TYPE \
    --patience $PATIENCE \
    --num_channels $NUM_CHANNELS \
    --num_classes $NUM_CLASSES \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS

echo $(date)