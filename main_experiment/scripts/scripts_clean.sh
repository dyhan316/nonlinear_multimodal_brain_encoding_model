#srun --nodes 1  --nodelist=node1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=12:00:00 --pty bash -i
#srun --nodes 1  --nodelist=node4 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=30:00:00 --pty bash -i
MODEL_LIST=("babylm/babyllama-100m-2024")

base_path=/scratch/connectome/ahhyun724/NeuroBabyLM/ #! change this to your own path
cd "${base_path}/nonlinear_multimodal_brain_encoding_model/main_experiment/" ; conda activate /scratch/connectome/ahhyun724/.conda/envs/NeurobabyLM_copy


# Step 1: Extract features
# you can either run in login node or interactive node(see srun command below, can change to node3)
#srun --nodes 1  --nodelist=node1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=30:00:00 --pty bash -i
for MODEL in "${MODEL_LIST[@]}"; do
    for STORY_IDX in {0..5}; do
        echo "🍏Extracting features for model: $MODEL, story partition: $STORY_IDX"
        python extract_and_save_LLAMA_features.py --model_name "$MODEL" --story_partition "$STORY_IDX"
    done
   echo "🍎Completed feature extraction for model: $MODEL"
done




# Step 2: Train + Evaluate encoder models
# you can either run in login node or interactive node(see srun command below, can change to node4)
#srun --nodes 1  --nodelist=node4 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=30:00:00 --pty bash -i

base_path=/scratch/connectome/ahhyun724/NeuroBabyLM/ #! change this to your own path
cd "${base_path}/nonlinear_multimodal_brain_encoding_model/main_experiment/" ; conda activate /scratch/connectome/ahhyun724/.conda/envs/NeurobabyLM_copy

MODEL_LAYER_LIST=("babylm/ltgbert-100m-2024_0" "babylm/ltgbert-100m-2024_4" "babylm/ltgbert-100m-2024_8" "babylm/ltgbert-100m-2024_11")
MODEL_LAYER_LIST=("babylm/ltgbert-10m-2024_0" "babylm/ltgbert-10m-2024_4" "babylm/ltgbert-10m-2024_8" "babylm/ltgbert-10m-2024_11")
MODEL_LAYER_LIST=("huggyllama/llama-7b_0" "huggyllama/llama-7b_8" "huggyllama/llama-7b_16" "huggyllama/llama-7b_24" "huggyllama/llama-7b_31")
MODEL_LAYER_LIST=("babylm/t5-base-strict-2023_0" "babylm/t5-base-strict-2023_4")
MODEL_LAYER_LIST=("babylm/t5-base-strict-2023_8" "babylm/t5-base-strict-2023_11")
MODEL_LAYER_LIST=("babylm/t5-base-strict-small-2023_0" "babylm/t5-base-strict-small-2023_4" "babylm/t5-base-strict-small-2023_8" "babylm/t5-base-strict-small-2023_11")
MODEL_LAYER_LIST=("EleutherAI/pythia-160m_0" "EleutherAI/pythia-160m_4" "EleutherAI/pythia-160m_8" "EleutherAI/pythia-160m_11")

for MODEL in "${MODEL_LAYER_LIST[@]}"; do
    for SUB in S1 S2 S3; do
        echo "🍏Training for model: $MODEL, subject: $SUB"
        python train_EM_for_main.py --subject "$SUB" --stim_setting "$MODEL" --enc_setting default --resp_setting default
        echo "🍏Evaluating for model: $MODEL, subject: $SUB"
        python evaluate_encoder.py --subject "$SUB" --stim_setting "$MODEL" --enc_setting default --resp_setting default
        echo "🍎Completed training and evaluation for model: $MODEL, subject: $SUB"
    done
done

#python train_EM_for_main.py --subject "$SUB" --stim_setting "$MODEL" --enc_setting default --resp_setting default
        
# Step 3: Summarize results
python analyze_result/run_extract.py --date 1207