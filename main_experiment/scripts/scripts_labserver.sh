#! NOT WORKING MODELS
#! step 1: error in loading models
 #"huggyllama/llama-7b" "EleutherAI/pythia-14m" "babylm/babyllama-10m-2024"
MODEL_LIST=("babylm/roberta-base-strict-2023" "babylm/roberta-base-strict-small-2023" "babylm/flamingo-2024") #! step 1: Token indices sequence length is longer than the specified maximum sequence length for this model (2426 > 128). Running this sequence through the model will result in indexing errors
MODEL_LIST=("babylm/flamingo-2024") #! step 1: got RuntimeError : Could not infer dtype of NoneType. //cf) output was generated regardless of Error
MODEL_LIST=("babylm/t5-base-strict-2023" "babylm/t5-base-strict-small-2023") #! step 1: ValueError: Unrecognized configuration class <class 'transformers.models.t5.configuration_t5.T5Config'> for this kind of AutoModel: AutoModelForCausalLM.
MODEL_LIST=("BabyLM-community/babylm-baseline-10m-gpt2" "BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus" "BabyLM-community/babylm-baseline-10m-gpt-bert-mixed")


#! WORKING MODELS (Running for 1.feature extraction)
MODEL_LIST=("babylm/babyllama-100m-2024") #! Worked, but suspicious. Token indices sequence length is longer than the specified maximum sequence length for this model (2426 > 128). Running this sequence through the model will result in indexing errors
MODEL_LIST=("babylm/opt-125m-strict-2023" "babylm/opt-125m-strict-small-2023") #! Sucessful!
MODEL_LIST=("babylm/ltgbert-100m-2024" "babylm/ltgbert-10m-2024") #! Succesful!
MODEL_LIST=("EleutherAI/pythia-160m")

#MODEL_LAYER_LIST=("huggyllama/llama-7b_12") #"babylm/babyllama-10m-2024_8") #"EleutherAI/pythia-14m_3") #"huggyllama/llama-7b_12"  
MODEL_LAYER_LIST=("EleutherAI/pythia-14m_2" "EleutherAI/pythia-14m_4" "EleutherAI/pythia-14m_5") #"huggyllama/llama-7b_12") #"babylm/babyllama-10m-2024_8") #"EleutherAI/pythia-14m_3") #"huggyllama/llama-7b_12"
MODEL_LAYER_LIST=("huggyllama/llama-7b_0" "huggyllama/llama-7b_8")
MODEL_LAYER_LIST=("huggyllama/llama-7b_8" "huggyllama/llama-7b_16" "huggyllama/llama-7b_24" "huggyllama/llama-7b_31")

MODEL_LAYER_LIST=("babylm/opt-125m-strict-2023_0" "babylm/opt-125m-strict-2023_4" "babylm/opt-125m-strict-2023_8" "babylm/opt-125m-strict-2023_11")
MODEL_LAYER_LIST=("babylm/opt-125m-strict-small-2023_0" "babylm/opt-125m-strict-small-2023_4" "babylm/opt-125m-strict-small-2023_8" "babylm/opt-125m-strict-small-2023_11")
MODEL_LAYER_LIST=("babylm/babyllama-100m-2024_0" "babylm/babyllama-100m-2024_4" "babylm/babyllama-100m-2024_8" "babylm/babyllama-100m-2024_12" "babylm/babyllama-100m-2024_15")

#! needs debugging
MODEL_LAYER_LIST=("babylm/flamingo-2024_0" "babylm/flamingo-2024_4" "babylm/flamingo-2024_8" "babylm/flamingo-2024_11")

#! Start from here!!
#srun --nodes 1  --nodelist=node1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=12:00:00 --pty bash -i
#srun --nodes 1  --nodelist=node4 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3G --time=30:00:00 --pty bash -i

base_path=/scratch/connectome/ahhyun724/NeuroBabyLM/ #! change this to your own path
cd "${base_path}/nonlinear_multimodal_brain_encoding_model/main_experiment/" ; conda activate /scratch/connectome/ahhyun724/.conda/envs/NeurobabyLM_copy


MODEL_LIST=("EleutherAI/pythia-70m" "EleutherAI/pythia-1b" "EleutherAI/pythia-2.8b")
MODEL_LIST=("EleutherAI/pythia-6.9b") #! 
MODEL_LIST=("EleutherAI/pythia-2.8b" "EleutherAI/pythia-6.9b")
MODEL 70M, 1B(same layer with babyllama), 2.8B(sth inbetween), 6.9B(same size with llama7B)


MODEL_LAYER_LIST=("EleutherAI/pythia-70m_5")
MODEL_LAYER_LIST=("EleutherAI/pythia-1b_0" "EleutherAI/pythia-1b_4" "EleutherAI/pythia-1b_8" "EleutherAI/pythia-1b_12" "EleutherAI/pythia-1b_15")
MODEL_LIST=("EleutherAI/pythia-2.8b_0" "EleutherAI/pythia-2.8b_8" "EleutherAI/pythia-2.8b_16" "EleutherAI/pythia-2.8b_24" "EleutherAI/pythia-2.8b_31")
MODEL_LIST=("EleutherAI/pythia-6.9b_0" "EleutherAI/pythia-6.9b_8" "EleutherAI/pythia-6.9b_16" "EleutherAI/pythia-6.9b_24" "EleutherAI/pythia-6.9b_31")


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


# MODEL_LAYER_LIST=("huggyllama/llama-7b_0" "huggyllama/llama-7b_8" "huggyllama/llama-7b_16" "huggyllama/llama-7b_24" "huggyllama/llama-7b_31")

# MODEL_LAYER_LIST=("")

# MODEL_LAYER_LIST=("babylm/ltgbert-100m-2024_0" "babylm/ltgbert-100m-2024_4" "babylm/ltgbert-100m-2024_8" "babylm/ltgbert-100m-2024_11")
# MODEL_LAYER_LIST=("babylm/ltgbert-10m-2024_0" "babylm/ltgbert-10m-2024_4" "babylm/ltgbert-10m-2024_8" "babylm/ltgbert-10m-2024_11")

MODEL_LAYER_LIST=("huggyllama/llama-7b_0" "huggyllama/llama-7b_8" "huggyllama/llama-7b_16" "huggyllama/llama-7b_24" "huggyllama/llama-7b_31")
MODEL_LAYER_LIST=("babylm/t5-base-strict-2023_0" "babylm/t5-base-strict-2023_4")
MODEL_LAYER_LIST=("babylm/t5-base-strict-2023_8" "babylm/t5-base-strict-2023_11")
MODEL_LAYER_LIST=("babylm/t5-base-strict-small-2023_0" "babylm/t5-base-strict-small-2023_4" "babylm/t5-base-strict-small-2023_8" "babylm/t5-base-strict-small-2023_11")

MODEL_LAYER_LIST=("EleutherAI/pythia-160m_0" "EleutherAI/pythia-160m_4" "EleutherAI/pythia-160m_8" "EleutherAI/pythia-160m_11")

MODEL=${MODEL_LAYER_LIST[0]}
#MODEL="babylm/opt-125m-strict-2023_8"

for SUB in S2 S3; do
        echo "🍏Training for model: $MODEL, subject: $SUB"
        python train_EM_for_main.py --subject "$SUB" --stim_setting "$MODEL" --enc_setting default --resp_setting default
        echo "🍏Evaluating for model: $MODEL, subject: $SUB"
        python evaluate_encoder.py --subject "$SUB" --stim_setting "$MODEL" --enc_setting default --resp_setting default
        echo "🍎Completed training and evaluation for model: $MODEL, subject: $SUB"
done


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