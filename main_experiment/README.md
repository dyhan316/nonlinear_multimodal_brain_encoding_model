# Aligning the Brain with Language Models Through a Nonlinear and Multimodal Approach

```
#salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu -c 64 --mem 100GB --account=m4135
salloc --nodes 1 --qos shared_interactive --time 04:00:00 --constraint cpu -c 64 --mem 100GB --account=m4135

module load conda 
conda activate DIVER_test

cd /global/cfs/cdirs/m4727/DANNY_EVCUATE_SEMANTIC_DECODING_FROM_NERSC_FOR_A_WHILE/semantic_decoding_neurobaby_lm_ver/nonlinear_multimodal_brain_encoding_model/main_experiment

python train_EM_for_main.py --subject S1 --stim_setting LLAMA7B_12 --enc_setting default --resp_setting PCA_10


python evaluate_encoder.py --subject S1 --stim_setting LLAMA7B_12 --enc_setting default --resp_setting PCA_10 --initial_trim 40



#* below : default, takes too long...
python train_EM_for_main.py --subject S1 --stim_setting LLAMA7B_12 --enc_setting default --resp_setting default

python extract_and_save_LLAMA_features.py --model_name babylm/babyllama-10m-2024 --story_partition 0 


```



This repository contains the code for the paper "Aligning the Brain with Language Models Through a Nonlinear and Multimodal Approach". The code implements methods for encoding brain activity using language models through nonlinear and multimodal approaches. This code is loosely based on [Scaling laws for language encoding models in fMRI](https://arxiv.org/abs/2305.11863) ([GitHub repository](https://github.com/HuthLab/encoding-model-scaling-laws)).
## Setup

### Prerequisites
* Python 3.7+
* PyTorch
* Transformers
* NumPy, SciPy, and other scientific computing libraries

### Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   conda env create -f env.yml
   ```
3. Add your HuggingFace login token in `LLM.py` (line 12)
4. Update directories in `config.py` to match your environment

### Data

The dataset is available for download from [the UT Austin Box repository](https://utexas.box.com/v/EncodingModelScalingLaws). This dataset contains brain responses to natural language stimuli, including fMRI recordings during story listening tasks, which are essential for training and evaluating the encoding models in this repository.

After downloading, place the data in the directory specified in `config.py` as `DATA_DIR`. The dataset includes:
- fMRI responses for multiple subjects
- Story text transcripts
- Audio files for speech stimuli

## Pipeline


### 0. Prepare PCA Components for Brain Responses

Before running any models, prepare PCA components for dimensionality reduction:

```bash
python PCA_exp/run_PCA_large.py
```

**Note:** Before running this script, you must modify the hardcoded paths in the file to match your environment:

```python
# In PCA_exp/run_PCA_large.py, change these lines:
sub = "S1"  # Change to appropriate subject
h5_folder_dir = f"/XXX/train_response/{sub}"  # Change to your path
```

This script:
- Concatenates all fMRI data across *train* stories
- Applies PCA to reduce dimensionality of brain responses
- Saves PCA components for later use in training/evaluation
- Required when using PCA-based response settings (e.g., PCA_512)


### 1. Extract Language Model Features

First, extract features from language models using:

```bash
python extract_and_save_LLAMA_features.py --model_name MODEL_NAME --story_partition PARTITION_INDEX
```

Arguments:
- `--model_name`: HuggingFace model path (e.g., "meta-llama/Meta-Llama-3-8B")
- `--story_partition`: Integer indicating which subset of stories to process (0-5).

The script will save extracted features to the directory specified in `config.py` as `STIM_FEATURE_EXTRACTED_DIR`. Features are saved as joblib files with layer-specific features.

### 2. Train Encoding Model

Train the brain encoding model with:

```bash
python train_EM_for_main.py --subject SUBJECT_ID --stim_setting STIMULUS_SETTING --enc_setting ENCODER_SETTING --resp_setting RESPONSE_SETTING
```

Arguments:
- `--subject`: Subject identifier (e.g., "S1")
- `--stim_setting`: Stimulus feature configuration
- `--enc_setting`: Encoder architecture
- `--resp_setting`: Response processing settings
- `--stim_setting_2`: Optional secondary stimulus setting for multimodal approaches

This script:
- Extracts audio features automatically if not already present
- For non-default encoder (i.e. non-linear) settings, performs hyperparameter optimization using Optuna
- Saves the trained encoding model in the `MODEL_DIR/subject` directory as `encoding_model_*.pth` (for neural networks) or `.npz` files (for ridge regression)
- For ridge regression, saves weights, alpha values, selected voxels, and training stories

### 3. Evaluate Encoding Model

Evaluate the trained model's performance:

```bash
python evaluate_encoder.py --subject SUBJECT_ID --stim_setting STIMULUS_SETTING --enc_setting ENCODER_SETTING --resp_setting RESPONSE_SETTING --initial_trim 40
```

Arguments match the training script, with:
- `--initial_trim`: Timepoints to trim from beginning (set to 40, in line with [Scaling laws for language encoding models in fMRI](https://arxiv.org/abs/2305.11863) for exact replication)

This script:
- Loads trained models and (separate) test data
- Generates predictions on test stories
- Calculates various performance metrics ($R^2$, correlations, $CC_{norm}$, ...)
- Saves results to `SCORE_DIR/subject_encoding_performance.jbl` including:
  - Voxel-wise correlations and prediction errors
  - Story-wise performance metrics
  - Model predictions for visualization
  - Configuration settings used for the evaluation

## Key Components

- **Language Models**: LLaMA models of various sizes extract semantic features from text
- **Audio Models**: Whisper models extract features from speech audio
- **Encoding Models**: Nonlinear neural networks (MLPs, Transformers) map model features to brain activity
- **PCA**: Dimensionality reduction for brain responses
- **Ridge Regression**: Baseline linear regression approach

## Project Structure

- `config.py`: Configuration settings and paths
- `extract_and_save_LLAMA_features.py`: Extract LLaMA features from text
- `train_EM_for_main.py`: Train the encoding model
- `evaluate_encoder.py`: Evaluate model performance
- `utils_*.py`: Utility functions for different components
- `DNN`: Deep neural network implementation
- `ridge_utils`: Ridge regression utilities
- `PCA_exp`: PCA experiments

## Advanced Usage

You can look at the code below for shell execution.

```bash
bash shell/run_whole_FINAL.sh
```

## Notes

- Processing large language models requires significant computational resources
- Set appropriate paths in config.py before running
