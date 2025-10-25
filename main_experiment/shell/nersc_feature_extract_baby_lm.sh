#!/bin/bash
#SBATCH --job-name=feature-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --time=01:00:00
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m4135
#SBATCH --output=feature_extract_slurm-%j.out
#SBATCH --error=feature_extract_slurm-%j.err

#* may not work! not tested

cd .. 
module load conda 
conda activate DIVER_finetuning

feature_extract() {
    local model_name="${1:-}"
    local story_partition="${2:-}"
    if [[ -z "$model_name" || -z "$story_partition" ]]; then
        echo "Usage: feature_extract <model_name> <story_partition>" >&2
        return 1
    fi

    srun python extract_and_save_LLAMA_features.py \
        --model_name "$model_name" \
        --story_partition "$story_partition"
}

MODEL_NAME="babylm/babyllama-10m-2024"
for PART in 0 1 2 3 4 5; do
    feature_extract "$MODEL_NAME" "$PART" &
done
wait



# # Auto-run if two args are provided to the script
# if [[ $# -ge 2 ]]; then
#     feature_extract "$1" "$2"
# fi
