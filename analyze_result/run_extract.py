import os
import re
from glob import glob
import pandas as pd
import joblib
import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, default='1027', help='date of the experiment results to analyze')
args = parser.parse_args()

# exp_result_dir = "/path/to/exp_results"
# exp_result_dir = "/alternative/path/to/exp_results"
exp_result_dir = '/scratch/connectome/ahhyun724/NeuroBabyLM/data_and_weights/exp_results' #! TODO : change this to your own path
score_folder_paths = glob(exp_result_dir + "/*/*/*/scores")

sub_list = ["S1", "S2", "S3"]
all_results = pd.DataFrame()

def process_folder(folder_dir):
    results = {}
    if len(os.listdir(folder_dir)) != 3:
        print(f"not enough! not all scores were calculated, has {len(os.listdir(folder_dir))} scores {folder_dir}")
        return None

    exp_info = folder_dir.split('/')[-3:-1]
    exp_info = "---".join(exp_info)

    print(exp_info)
    # Load all data once
    all_data = {sub: joblib.load(f'{folder_dir}/{sub}_encoding_performance.jbl') for sub in sub_list}
    
    # Assuming the configuration is the same for all subjects, load it from the first subject
    data = all_data[sub_list[0]]

    # Process each metric
    for k in ['mean_SPE', 'mean_cc_norm', 'mean_cc_max', 'mean_corrs_unnorm', 'mean_r', 'mean_r2']:
        # Extract scores for all subjects for the current metric
        total_perf_list = [all_data[sub][k] for sub in sub_list]
        
        # Calculate mean and standard deviation
        results[(exp_info, k + '_mean')] = np.mean(total_perf_list)
        results[(exp_info, k + '_std')] = np.std(total_perf_list)

    # Process storywise metrics
    for k in ['mean_storywise_r', 'mean_storywise_r2']:
        total_perf = {}
        for sub in sub_list:
            storywise_sub_score = all_data[sub][k]
            for key, value in storywise_sub_score.items():
                if key in total_perf:
                    total_perf[key] += value
                else:
                    total_perf[key] = value

        for s in total_perf.keys():
            results[(exp_info, s + k[14:] + '_mean')] = total_perf[s] / len(sub_list)
    
    # Save the sub paths and cc_norm, corrs_unnorm
    for sub in sub_list:
        results[(exp_info, f'{sub}_score_paths')] = f'{folder_dir}/{sub}_encoding_performance.jbl'
    
    # Separate numeric and string results
    numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    string_results = {k: v for k, v in results.items() if isinstance(v, str)}

    # Create DataFrame for numeric results
    tmp_numeric = pd.DataFrame.from_dict(data=numeric_results, orient='index').reset_index()
    tmp_numeric.columns = ['index', 'value']
    df_numeric = pd.DataFrame(tmp_numeric['index'].tolist(), index=tmp_numeric.index, columns=['exp_info', 'metrics'])
    df_numeric['value'] = tmp_numeric['value']
    df_numeric = df_numeric.pivot_table(index='exp_info', columns='metrics', values='value').reset_index()

    # Create DataFrame for string results
    tmp_string = pd.DataFrame.from_dict(data=string_results, orient='index').reset_index()
    tmp_string.columns = ['index', 'value']
    df_string = pd.DataFrame(tmp_string['index'].tolist(), index=tmp_string.index, columns=['exp_info', 'metrics'])
    df_string['value'] = tmp_string['value']
    df_string = df_string.pivot_table(index='exp_info', columns='metrics', values='value', aggfunc=lambda x: ' '.join(x)).reset_index()

    # Merge numeric and string DataFrames
    df = pd.merge(df_numeric, df_string, on='exp_info', how='outer')

    for c in ['stimulus_setting', 'encoding_setting', 'response_setting', 'name_setting', 'task', 'format_name_name']:
        df[c] = data['config'][c]

    if 'stimulus_setting_2' in data['config']:
        df['stimulus_setting_2'] = data['config']['stimulus_setting_2']

    return df


# Use ProcessPoolExecutor to parallelize folder processing
with ProcessPoolExecutor(max_workers = 16) as executor:
    results = list(tqdm.tqdm(executor.map(process_folder, score_folder_paths), total=len(score_folder_paths), desc="Processing score folders in parallel"))

# Filter out None results and concatenate DataFrames
results = [r for r in results if r is not None]
if results:
    all_results = pd.concat(results)

all_results = pd.concat(results)
all_results.to_csv(f"all_results_{args.date}.csv", index=False)
