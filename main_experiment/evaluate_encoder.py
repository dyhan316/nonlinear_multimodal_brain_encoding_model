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

from DNN import DNN_main
from utils_eval import spe_and_cc_norm, compute_r_single_trial, plot_corrs_all_trials, config_into_dict_for_saving
from utils_enc_model import get_model_pred, get_vox_pred_from_model_pred
import torch 

### Ignore irrelevant warnings that muck up the notebook
import sys 
import warnings

import joblib 
from train_EM_for_main import legacy_when_default_stim

if not sys.warnoptions:
    warnings.simplefilter("ignore") 
    
np.random.seed(42)

def reproduce_OPT_result(args) : 
    import joblib 
    import ridge_utils
    from ridge_utils.util import make_delayed
    from ridge_utils import npp 
    thing_path = '/path/to/sample_data/wheretheressmoke_preds'

    args.subject = "S3"
    #custom sub, reload the response
    
    #reset the subject
    tensession_test_resp = get_test_resp("tensessions", args.subject, None, None, **config.R_SETTINGS) #experiment, task : None cuz not needed.. (only evaluate tensession from perceived speech)

    features_dict = joblib.load(f'{thing_path}/downsampled_featureseqs_opt33b_layer33.jbl')
    # Trim and zscore the model features, concat test features
    trim_start = 50 ;  trim_end = 5 

    Tstim_dict  = {test_story : np.nan_to_num(npp.zs(features_dict[test_story][10:-5])) for test_story in stories} #10,-5 : manual traim 
    sub = args.subject
    weights = joblib.load(f"{thing_path}/{sub}_opt33b_wts_layer33.jbl")
    resp_dict = joblib.load(f"{thing_path}/UTS0{sub[-1]}_responses.jbl")
    resp_dict = {story : np.nan_to_num(resp_dict[story]) for story in stories}

    avg_test_resp_dict = resp_dict
    model_vox_pred_dict = {story : np.dot(make_delayed(Tstim_dict[story], range(1,5)),  weights) for story in stories}

    return tensession_test_resp, avg_test_resp_dict, model_vox_pred_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--stim_setting", type = str, default = "default")
    parser.add_argument("--stim_setting_2", type = str, default = None) 
    parser.add_argument("--enc_setting", type = str, default = "default")
    parser.add_argument("--resp_setting", type = str, default = "default")
    parser.add_argument("--initial_trim", type = int, default = 40)
    parser.add_argument("--name", type = str, default = "default_name")
    parser.add_argument("--OPT_reproduction", type = bool, default = False) 
    
    args = parser.parse_args()

    config.initialize(args.stim_setting, args.stim_setting_2, args.enc_setting, args.resp_setting, args.name, args.subject) 
    
    #! remove later when evaluating with stim_2 is implemented!
    if args.stim_setting_2 is not None : 
        pass
        # raise NotImplementedError("evaluating with rstim2not implmemented yet, as get_model_pred is currently only made for single input stimulus!")
    
    if "PCAonIMP" == args.resp_setting.split("_")[0]:
        raise ValueError("PCAonIMP cannot be used to calculate encoder error, as it cannot give me the voxelwise values!")
    elif args.resp_setting.startswith("default_") :
        raise ValueError(f"{args.resp_setting} does not need to be done, as this is no different from default ")
    
    stories = config.test_list #test story
    
    ##*1. response : two types... 1. tensession (10 trials of wherehteressmoke) ==> for SPE and so on 2. test_resp => for aveareg r^2 values (comparison)
    tensession_test_resp = get_test_resp("tensessions", args.subject, None, None, **config.R_SETTINGS) #experiment, task : None cuz not needed.. (only evaluate tensession from perceived speech)
    avg_test_resp_dict = get_test_resp("avg_test_resp", args.subject, None, None, stories = stories) #preaveraged test response provided by encoding scaling law paper
    # tensession_test_resp : (10, 291, 95556) (i.e. number of trials)
    
    if args.subject == "S1": avg_test_resp_dict['wheretheressmoke'] = tensession_test_resp.mean(axis = 0)
    
    ##*2. stimulus (i.e. LLM embeddings of input features)
    # #TODO : may be needed to be changed once not doing resampling or sth (i.e. if stim_setting is not default)
    if args.stim_setting == "default":
        rstim, rstim_2 = legacy_when_default_stim(args, stories, stacked = False) #always stack false 
    else :  #use pretrained stim features! 
        rstim, _ , _ = load_stim_features(args.stim_setting, config.S_SETTINGS, stories, 
                                          config.S_SETTINGS['FEATURE_LAYER'], tr_stats = None, stacked = False, downsampled = True)
        rstim_2, _ , _ = load_stim_features(args.stim_setting_2, config.S_SETTINGS_2, stories, 
                                          config.S_SETTINGS['FEATURE_LAYER'], tr_stats = None, stacked = False, downsampled = True) if args.stim_setting_2 is not None else (None, None, None)
    
    ##*3. prediction (i.e. encoding_model(stimulus))
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    if config.E_SETTINGS['save_type'] == "numpy": 
        encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % args.gpt)) 
        encoding_model_dict = {key : encoding_model[key] for key in encoding_model.files} #make into dict (to put in get_model_pred) 
    elif config.E_SETTINGS['save_type'] == "torch": 
        encoding_model_dict = torch.load(os.path.join(load_location, "encoding_model_%s.pth" % args.gpt)) # Already dict (e.g., torch.load returns dict directly)
    
    model_vox_pred_dict = {} #for each story
    for story in stories:
        # 3-2. Run model prediction (eval). (fold-wise results & ensembling of folds results)
        if args.enc_setting.startswith('MultimodalMLP') : 
            model_input = [rstim[story], rstim_2[story]]
        else: 
            model_input = rstim[story] if args.stim_setting_2 is None else config.E_SETTINGS['input_combiner'](rstim[story], rstim_2[story]) if args.enc_setting != "default" else np.hstack([rstim[story], rstim_2[story]]) ##if rstim_2 is None, then just use rstim
        model_target = avg_test_resp_dict[story] #only used in transformers, but still pass it!
        model_foldwise_pred, ensemble_model_pred = get_model_pred(args.enc_setting, model_input, config.E_SETTINGS['validation_method'], model_target, **encoding_model_dict) 
        #! TODO : this part (get_model_pred function) is only made for single stimulus thing!! mulit-stimulus thing is not implemented yet! (i.e. rstim_2 is not used)
        
        # 3-3. Getting FINAL model_pred (i.e. aggregating foldwise predictions)
        if "default" == args.enc_setting: final_model_pred = model_foldwise_pred[0] #if default, just take the first one (no need to do ensemble)
        elif "MLP_" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        elif "MLPTanh_" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        elif "reducedLinear_" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        elif "single" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        elif "MultimodalMLP" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        elif "MultimodalMLPTanh" in args.enc_setting: final_model_pred = ensemble_model_pred['mean']
        else : raise ValueError(f"enc_setting {args.enc_setting} not implemented yet")    
        
        # 3-4. final_model_pred =(reverse PCA or sth)=> model_vox_pred (i.e. convert model's prediction to voxel prediction)
        model_vox_pred = get_vox_pred_from_model_pred(args.resp_setting, final_model_pred, args.subject)
        
        model_vox_pred_dict[story] = model_vox_pred
    
    #if OPT reproduction mode, overwrite resp and predictions with OPT results
    if args.OPT_reproduction : 
        config.SCORE_DIR = os.path.join(config.REPO_DIR, "exp_results", "OPT_REPRODUCTION" ) #rewrite to put it at OPT reproduction thing 
        tensession_test_resp, avg_test_resp_dict, model_vox_pred_dict = reproduce_OPT_result(args)
        
    ##*4. Trimming 
    adjusted_trimming_amount = args.initial_trim - (model_target.shape[0]-model_foldwise_pred[0].shape[0]) #only uses last story to get the trim amount, but doens't matter 
    
    #trimming tensession thing AND avg test resp and its prediction
    tensession_test_resp = tensession_test_resp[:,args.initial_trim:,:] 
    avg_test_resp_dict = {story : avg_test_resp_dict[story][args.initial_trim:,:] for story in stories}
    model_vox_pred_dict = {story : model_vox_pred_dict[story][adjusted_trimming_amount:,:] for story in stories}


    ##*5. Calculating performance (calculate SPE and similar metrics using tensessions, and r^2 values using avg_test_resp)
    #5-1. computing SPE and cc (voxel wise) (tensession thing, for wheretheressmoke thing )
    
    # for getting the full cc_max thing!
    if args.stim_setting == "default": # This is the same regardless of stim and encoder settings, so let's do it this way!
        save_path = f"/path/to/analysis_data/full_cc_max_{args.subject}.npy"
        SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(tensession_test_resp, model_vox_pred_dict['wheretheressmoke'], data_norm = False, max_flooring=0.0)
        np.save(save_path, cc_max)        
        import pdb ; pdb.set_trace()
    SPE, cc_norm, cc_max, corrs_unnorm = spe_and_cc_norm(tensession_test_resp, model_vox_pred_dict['wheretheressmoke'], data_norm = False, max_flooring=0.25) #max flooring : as said in the paper.. also data_norm False cuz it doesn't affect and better to do just in case 
    
    #5-2. computing r and r^2 values
    concat_avg_resp, concat_model_vox_pred = np.concatenate([avg_test_resp_dict[story] for story in stories]), np.concatenate([model_vox_pred_dict[story] for story in stories])
    average_resp_r_list = compute_r_single_trial(concat_avg_resp, concat_model_vox_pred)
    average_resp_r2_list = np.abs(average_resp_r_list)*average_resp_r_list #not r^2 cuz this is the method that huth used (i.e. r^2 = |r|*r)

    #5-3. computing r and r^2 values for each story
    storywise_average_resp_r_list = {story : compute_r_single_trial(avg_test_resp_dict[story],model_vox_pred_dict[story]) for story in stories}
    storywise_average_resp_r2_list = {story : np.abs(storywise_average_resp_r_list[story])*storywise_average_resp_r_list[story] for story in stories} #not r^2 cuz this is the method that huth used

    
    ##*6. save results
    save_location = os.path.join(config.SCORE_DIR )#, args.subject, "enc_performance")
    os.makedirs(save_location, exist_ok=True) 
    
    # Prepare the data to be saved
    data_to_save = {
        # tensession stats (5-1)
        "SPE": SPE, "cc_norm": cc_norm, "cc_max": cc_max, "corrs_unnorm": corrs_unnorm, 
        "mean_SPE": SPE.mean(), "mean_cc_norm": cc_norm.mean(), "mean_cc_max": cc_max.mean(), "mean_corrs_unnorm": corrs_unnorm.mean(),
        
        # average response stats (5-2)
        "average_resp_r_list": average_resp_r_list, "average_resp_rsq_list": average_resp_r2_list,
        "mean_r": average_resp_r_list.mean(), "mean_r2": average_resp_r2_list.mean(),
        
        # storywise response stats (5-3)
        "storywise_average_resp_r_list": storywise_average_resp_r_list, 
        "storywise_average_resp_rsq_list": storywise_average_resp_r2_list,
        "mean_storywise_r": {story: np.nan_to_num(storywise_average_resp_r2_list[story]).mean() for story in stories}, 
        "mean_storywise_r2": {story: np.nan_to_num(storywise_average_resp_r2_list[story]).mean() for story in stories},
        
        # saving the predictions (model_vox_pred_dict)
        "model_vox_pred_dict": model_vox_pred_dict,
        
        
        # saving config and other misc stuff
        "config": config_into_dict_for_saving(config),
        "config_R_SETTINGS": config_into_dict_for_saving(config.R_SETTINGS),
        "config_E_SETTINGS": config_into_dict_for_saving(config.E_SETTINGS),
        "config_S_SETTINGS": config_into_dict_for_saving(config.S_SETTINGS),
        "config_S_SETTINGS_2": config_into_dict_for_saving(config.S_SETTINGS_2),
        
        # Uncomment and include the following if needed
        # "trial_wise_r_list": trial_wise_r_list,
        # "trial_wise_rsq_list": trial_wise_rsq_list,
        # "trial_wise_mean_r": np.array([i.mean() for i in trial_wise_r_list]),
        # "trial_wise_mean_rsq": np.array([i.mean() for i in trial_wise_rsq_list]),
        # "all_trial_mean_r": np.array([i.mean() for i in trial_wise_r_list]).mean(),
        # "all_trial_mean_rsq": np.array([i.mean() for i in trial_wise_rsq_list]).mean(),
    }
    joblib.dump(data_to_save, save_location + f"/{args.subject}_encoding_performance.jbl")
