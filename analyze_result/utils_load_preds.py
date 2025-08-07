import joblib 
import numpy as np 
import os 
import sys 
sys.path.append('../main_experiment')
import config 
from utils_resp import get_test_resp
import matplotlib.pyplot as plt

from typing import List, Dict
from tqdm import tqdm

stories = config.test_list #test story
initial_trim = 40

def subset_dict_from_dict(dict_to_subset, keys_to_subset):
    return {key : dict_to_subset[key] for key in keys_to_subset}    


######====LOADING PREDICTIONS====######
def model_score_file_loader(sub, baseline_dir):
    return joblib.load(os.path.join(baseline_dir, f"{sub}_encoding_performance.jbl"))

def model_measure_loader(measure, sub, baseline_dir):
    return model_score_file_loader(sub, baseline_dir)[measure]

def model_vox_pred_loader(story, sub, baseline_dir):
    return model_measure_loader("model_vox_pred_dict", sub, baseline_dir)[story]

def get_model_pred_sub(sub, baseline_dir, stories_to_load = stories):
    r"""
    by default loads all test stories 
    """
    model_vox_pred_list = [model_vox_pred_loader(story, sub, baseline_dir) for story in stories_to_load] #getting summed model preds
    all_model_vox_pred = np.vstack(model_vox_pred_list) #stacking every story (for r, r2 calc)
    model_tensession_vox_pred = model_vox_pred_list[0] #just getting tensession predictions (for SPE, ...)
    return all_model_vox_pred, model_tensession_vox_pred

#* IMP : this is the final thing that we will use to load the shit!
def get_model_pred_allsub(name_path_dict : Dict, stories_to_load = stories, subs = ["S1", "S2", "S3"]):
    final_dict = {}
    for model_nickname, exp_results_dir in tqdm(name_path_dict.items()):
        total_subs_dict = {}
        for sub in subs:
            all_model_vox_pred, model_tensession_vox_pred = get_model_pred_sub(sub, exp_results_dir, stories_to_load)
            measures_dict = subset_dict_from_dict(model_score_file_loader(sub, exp_results_dir), ["cc_norm", "cc_max", "corrs_unnorm", "mean_cc_norm", "mean_r2"])
            total_subs_dict[sub] = {"all_model_vox_pred" : all_model_vox_pred, "model_tensession_vox_pred" : model_tensession_vox_pred, **measures_dict}
        final_dict[model_nickname] = total_subs_dict
    return final_dict    

######====LOADING ACTUAL RESPONSES====######
def get_test_resp_trimmed_sub(sub, stories_to_load = stories):
    avg_test_resp_dict = get_test_resp("avg_test_resp", sub, None, None, stories = stories_to_load) #preaveraged test response provided by encoding scaling law paper
    tensession_test_resp = get_test_resp("tensessions", sub, None, None) #experiment, task : None cuz not needed.. (only evaluate tensession from perceived speech)
    
    all_test_resp = np.vstack([avg_test_resp_dict[story][initial_trim:, :] for story in stories_to_load])
    final_tensession_test_resp = tensession_test_resp[:,initial_trim:, :]

    return all_test_resp, final_tensession_test_resp

def get_test_resp_trimmed_allsub(subs = ["S1", "S2", "S3"]):
    final_dict = {}
    for sub in tqdm(subs):
        all_test_resp, final_tensession_test_resp = get_test_resp_trimmed_sub(sub)
        final_dict[sub] = {"all_test_resp" : all_test_resp, "final_tensession_test_resp" : final_tensession_test_resp}
    return final_dict

