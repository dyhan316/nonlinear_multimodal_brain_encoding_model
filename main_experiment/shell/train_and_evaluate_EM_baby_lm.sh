#!/bin/bash
#SBATCH --job-name=S2_trainEM-baby-lm
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:20:00
#SBATCH -c 64
#SBATCH --mem=200GB
#SBATCH --account=m4135
#SBATCH --output=S2_trainEM_eval_baby_lm_slurm-%j.out
#SBATCH --error=S2_trainEM_eval_baby_lm_slurm-%j.err

###sdfkjsdlkfj##
#* may not work! not tested

cd .. 
module load conda 
conda activate DIVER_finetuning


train_eval_em() {
    local subject="$1"
    local stim_idx="$2"

    if [[ -z "$subject" || -z "$stim_idx" ]]; then
        echo "Usage: train_em <subject> <stim_index> [extra-args]" >&2
        return 1
    fi

    local stim_setting="babylm-babyllama-10m-2024_${stim_idx}"

    python train_EM_for_main.py \
        --subject "$subject" \
        --stim_setting "$stim_setting" \
        --enc_setting default \
        --resp_setting default
    
    python evaluate_encoder.py \
        --subject "$subject" \
        --stim_setting "$stim_setting" \
        --enc_setting default \
        --resp_setting default   
}


subjects=(S2) #(S1 S2 S3)
stim_indices=(1 3 5 7 9 11 13 15 17)

for subj in "${subjects[@]}"; do
    for idx in "${stim_indices[@]}"; do
        train_eval_em "$subj" "$idx"
    done
done




# feature_extract() {
#     local model_name="${1:-}"
#     local story_partition="${2:-}"
#     if [[ -z "$model_name" || -z "$story_partition" ]]; then
#         echo "Usage: feature_extract <model_name> <story_partition>" >&2
#         return 1
#     fi

#     srun python extract_and_save_LLAMA_features.py \
#         --model_name "$model_name" \
#         --story_partition "$story_partition"
# }

# MODEL_NAME="babylm/babyllama-10m-2024"
# for PART in 0 1 2 3 4 5; do
#     feature_extract "$MODEL_NAME" "$PART" &
# done
# wait



# # Auto-run if two args are provided to the script
# if [[ $# -ge 2 ]]; then
#     feature_extract "$1" "$2"
# fi

subjects=(S1) #(S1 S2 S3)
stim_indices=(1 3 5 7 9 11 13 15 17)

for subj in "${subjects[@]}"; do
    for idx in "${stim_indices[@]}"; do
        train_eval_em "$subj" "$idx"
    done
done
