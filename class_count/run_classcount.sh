#!/bin/bash

#SBATCH --gres gpu:1
#SBATCH --constraint a6000
#SBATCH --constraint m48
#SBATCH --mem 32G
#SBATCH --time 10:00:00
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
INPUT_DIR="/share/projects/siamdl/data/small/"
OUT_PATH="/share/projects/siamdl/outputs/${SLURM_JOBID}_classcount/"
INPUT_TYPE="s2"
BATCH_SIZE=16
PROCESS_LEVEL="l1c"
LEARN_TYPE="csl"
NUM_CLASSES=11

# Create output directory
mkdir -p $OUT_PATH
echo "Starting script"
echo $(date)

python class_count.py \
    --input_dir $INPUT_DIR \
    --out_path $OUT_PATH \
    --input_type $INPUT_TYPE \
    --batch_size $BATCH_SIZE \
    --process_level $PROCESS_LEVEL \
    --learn_type $LEARN_TYPE \
    --num_classes $NUM_CLASSES \

echo $(date)
