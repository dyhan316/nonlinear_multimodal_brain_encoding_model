#python evaluate_with_validation_mask.py --subject S1 --stim_setting LLAMA30B_18 --stim_setting_2 whisper-large_final_default_default --enc_setting MLP_64 --resp_setting PCA_512 --name "validation_method CV loss_criteria MAE multimodal_method concat_dataset"

import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
# from LLM import LLM 
from StimulusModel import LMFeatures
from utils_stim import get_stim, load_stim_features
from utils_resp import get_resp, get_test_resp
import tqdm #not used, but use later if needed
import wandb #not used, but use later if needed
import time  #not used, but use later if needed

from DNN import DNN_main
from utils_eval import spe_and_cc_norm, compute_r_single_trial, plot_corrs_all_trials, config_into_dict_for_saving
from utils_enc_model import get_model_pred, get_vox_pred_from_model_pred
import torch 

### Ignore irrelevant warnings that muck up the notebook
import sys 
import warnings

import joblib 
from train_EM_for_main import legacy_when_default_stim
import fnmatch

if not sys.warnoptions:
    warnings.simplefilter("ignore") # because np.sqrt(negative value) warnings appear during SPE calculation, but can be ignored (later changed to zero)
    
from config import format_name, reverse_format_name
np.random.seed(42)

def remove_multimodal_from_name(name):
    formatted_name = format_name(name)
    original_list = [i.split('__') for i in formatted_name.split('-')]
    filtered_list = [element for element in original_list if element[0] != 'multimodal_method']
    formatted_name = '-'.join(['__'.join(i) for i in filtered_list])
    single_modality_name = reverse_format_name(formatted_name)
    return single_modality_name

##===TODO=====##
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--stim_setting", type = str, default = "default")
    parser.add_argument("--stim_setting_2", type = str, default = None) 
    parser.add_argument("--enc_setting", type = str, default = "default")
    parser.add_argument("--resp_setting", type = str, default = "default")
    parser.add_argument("--initial_trim", type = int, default = 40) # trim first 40 timepoints: as done in scaling encoding paper
    parser.add_argument("--name", type = str, default = "default_name")
    
    #! specific to this code, since single modality model might have different enc_setting or name or resp_setting 
    parser.add_argument("--single_stim_setting", type = str, default = None)
    parser.add_argument("--single_enc_setting", type = str, default = None)
    parser.add_argument("--single_resp_setting", type = str, default = None)
    parser.add_argument("--single_name", type = str, default = None)
    
    args = parser.parse_args()

    #if single_* args are None, then just set them as same from the original args 
    if args.single_stim_setting is None: args.single_stim_setting = args.stim_setting
    if args.single_enc_setting is None: args.single_enc_setting = args.enc_setting
    if args.single_resp_setting is None: args.single_resp_setting = args.resp_setting
    if args.single_name is None: args.single_name = remove_multimodal_from_name(args.name) 
    
    #getting args.name with multimodality removed! 
    
    assert args.stim_setting_2 
    if "PCAonIMP" == args.resp_setting.split("_")[0]:
        raise ValueError("PCAonIMP cannot be used to calculate encoder error, as it cannot give me the voxelwise values!")
    elif args.resp_setting.startswith("default_") :
        raise ValueError(f"{args.resp_setting} does not need to be done, as this is no different from default (important voxels are only used for beam search, probability models, not encoding models)")
    
    # now getting the save_locations for both single and multimoidal things 
    config.initialize(args.single_stim_setting, None, args.single_enc_setting, args.single_resp_setting, args.single_name, args.subject)
    save_location_single = os.path.join(config.SCORE_DIR )
    model_save_location_single = os.path.join(config.MODEL_DIR, args.subject)
    
    config.initialize(args.stim_setting, args.stim_setting_2, args.enc_setting, args.resp_setting, args.name, args.subject) 
    save_location = os.path.join(config.SCORE_DIR )
    model_save_location = os.path.join(config.MODEL_DIR, args.subject)
    
    assert config.E_SETTINGS['combined_dataset_generator']  
    assert args.enc_setting != "default" #not done yet, since voxelwise validation is not implmented in default... just skip this!
    
    score_joblib_single = joblib.load(save_location_single + f"/{args.subject}_encoding_performance.jbl")
    model_pth_single = torch.load(model_save_location_single + f"/encoding_model_{args.gpt}.pth")
    score_joblib = joblib.load(save_location + f"/{args.subject}_encoding_performance.jbl")
    model_pth = torch.load(model_save_location + f"/encoding_model_{args.gpt}.pth")
    
    find_all_matching_keys = lambda model_pth, pattern : [key for key in model_pth.keys() if fnmatch.fnmatch(key, pattern)]

    #!======================================================================================================    
       
    # stacked_validation_corr_single = np.stack([model_pth_single[i] for i in find_all_matching_keys(model_pth_single, "true_voxelwise_corr_*")])
    # stacked_validation_corr = np.stack([model_pth[i] for i in find_all_matching_keys(model_pth, "true_voxelwise_corr_*")])
    
    stacked_validation_corr_single = np.stack([model_pth_single[i] for i in find_all_matching_keys(model_pth_single, "target_recon_voxelwise_corr_*")])
    stacked_validation_corr = np.stack([model_pth[i] for i in find_all_matching_keys(model_pth, "target_recon_voxelwise_corr_*")])
    
    # stacked_validation_corr_single = - np.stack([model_pth_single[i] for i in find_all_matching_keys(model_pth_single, "target_recon_voxelwise_mae_*")])
    # stacked_validation_corr =  - np.stack([model_pth[i] for i in find_all_matching_keys(model_pth, "target_recon_voxelwise_mae_*")])
    
    #!======================================================================================================
    mean_validation_corr_single = stacked_validation_corr_single.mean(axis = 0)
    mean_validation_corr = stacked_validation_corr.mean(axis = 0)
    
    
    
    ##get validation masked prediction (i.e. combine the two models' predictions based on the validation mask)
    model_vox_pred_dict_single = score_joblib_single['model_vox_pred_dict']
    model_vox_pred_dict = score_joblib['model_vox_pred_dict']
    aggregate_model_vox_pred_dict = {} #for each story, the aggregate model prediction (after validation masking between two models)
    
    mask_multimodal_is_better = mean_validation_corr >= mean_validation_corr_single
    for story in model_vox_pred_dict.keys():
        aggregate_model_vox_pred_dict[story] = np.where(mask_multimodal_is_better, model_vox_pred_dict[story], model_vox_pred_dict_single[story])
    
    ##getting resp things (to compare against and evaluate the aggregate encoder )
    tensession_test_resp = get_test_resp("tensessions", args.subject, None, None, **config.R_SETTINGS) #experiment, task : None cuz not needed.. (only evaluate tensession from perceived speech)
    avg_test_resp_dict = get_test_resp("avg_test_resp", args.subject, None, None, stories = config.test_list) #preaveraged test response provided by encoding scaling law paper
    tensession_test_resp = tensession_test_resp[:,args.initial_trim:,:] 
    avg_test_resp_dict = {story : avg_test_resp_dict[story][args.initial_trim:,:] for story in config.test_list}
    model_vox_pred_dict = {story : model_vox_pred_dict[story][args.initial_trim:,:] for story in config.test_list}
    
    
    ##get the performance of the aggregate model prediction
    #1. SPE and cc (voxel wise) (tensession thing, for wheretheressmoke thing )
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(tensession_test_resp, aggregate_model_vox_pred_dict['wheretheressmoke'], data_norm = False, max_flooring=0.25) #max flooring : as said in the paper.. also data_norm False cuz it doesn't affect and better to do just in case
    
    #2. r and r^2 values
    concat_avg_resp, concat_model_vox_pred = np.concatenate([avg_test_resp_dict[story] for story in config.test_list]), np.concatenate([aggregate_model_vox_pred_dict[story] for story in config.test_list])
    average_resp_r_list = compute_r_single_trial(concat_avg_resp, concat_model_vox_pred)
    average_resp_r2_list = np.abs(average_resp_r_list)*average_resp_r_list #not r^2 cuz this is the method that huth used (i.e. r^2 = |r|*r)
    
    #3. r and r^2 values for each story
    storywise_average_resp_r_list = {story : compute_r_single_trial(avg_test_resp_dict[story],aggregate_model_vox_pred_dict[story]) for story in config.test_list}
    storywise_average_resp_r2_list = {story : np.abs(storywise_average_resp_r_list[story])*storywise_average_resp_r_list[story] for story in config.test_list} #not r^2 cuz this is the method that huth used
    
    #4. save results
    print(np.sum(mask_multimodal_is_better)/mask_multimodal_is_better.shape[0],  average_resp_r2_list.mean(), score_joblib_single['mean_r2'], score_joblib['mean_r2'])
    print((score_joblib['average_resp_rsq_list'] > score_joblib_single['average_resp_rsq_list']).sum()/94251)
    #np.save(f'./{args.subject}_improvements.npy',(score_joblib['average_resp_rsq_list'] -score_joblib_single['average_resp_rsq_list']))

    
    
    