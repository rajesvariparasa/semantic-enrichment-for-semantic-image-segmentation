#!/bin/bash

#SBATCH --gres gpu:1
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
cd scripts
INPUT_DIR="/share/projects/siamdl/data/small/"
OUT_PATH="/share/projects/siamdl/outputs/${SLURM_JOBID}_predictonly/quickview/"
MODEL_PATH="/share/projects/siamdl/outputs/43450_20240418_132245/quickview/model_outputs/best_model.pth"
INPUT_TYPE="s2"
BATCH_SIZE=16
PROCESS_LEVEL="l1c"
LEARN_TYPE="csl"
NUM_CLASSES=11
GENERATE_CURVES=true
REMARKS="Predicting only. No training. Model used: <enter jobid of model used for prediction>"

# Create output directory
mkdir -p $OUT_PATH

# Save arguments to a file
echo "Arguments passed to the script:" > $OUT_PATH/arguments.txt
echo "Input Directory: $INPUT_DIR" >> $OUT_PATH/arguments.txt
echo "Output Path: $OUT_PATH" >> $OUT_PATH/arguments.txt
echo "Model Path: $MODEL_PATH" >> $OUT_PATH/arguments.txt
echo "Input Type: $INPUT_TYPE" >> $OUT_PATH/arguments.txt
echo "Batch Size: $BATCH_SIZE" >> $OUT_PATH/arguments.txt
echo "Process Level: $PROCESS_LEVEL" >> $OUT_PATH/arguments.txt
echo "Learn Type: $LEARN_TYPE" >> $OUT_PATH/arguments.txt
echo "Number of Classes: $NUM_CLASSES" >> $OUT_PATH/arguments.txt
echo "Generate Curves: $GENERATE_CURVES" >> $OUT_PATH/arguments.txt
echo "Remarks: $REMARKS" >> $OUT_PATH/arguments.txt

echo "Starting script"
echo $(date)

python predict_only.py \
    --input_dir $INPUT_DIR \
    --out_path $OUT_PATH \
    --model_path $MODEL_PATH \
    --input_type $INPUT_TYPE \
    --batch_size $BATCH_SIZE \
    --process_level $PROCESS_LEVEL \
    --learn_type $LEARN_TYPE \
    --num_classes $NUM_CLASSES \
    --generate_curves $GENERATE_CURVES

cd
cp "siamdl${SLURM_JOBID}.out" $OUT_PATH
echo $(date)
