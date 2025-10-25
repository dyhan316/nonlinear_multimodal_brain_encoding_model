import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
# from LLM import LLM 
from StimulusModel import LMFeatures, get_stim_features, get_stim_model
from utils_stim import get_stim, load_stim_features
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge
import tqdm #not used, but use later if needed
# import wandb #not used, but use later if needed
# import time  #not used, but use later if needed

from DNN import DNN_main
import joblib
np.random.seed(42)
def legacy_when_default_stim(args, stories ,stacked = True): 
    # load gpt (here GPT could be LLM or GPT)
    gpt = get_stim_model(args, config.S_SETTINGS['audio_or_LLM'], config.S_SETTINGS)
    gpt_2 = get_stim_model(args, config.S_SETTINGS_2['audio_or_LLM'], config.S_SETTINGS_2) if args.stim_setting_2 is not None else None
    features = get_stim_features(config.S_SETTINGS['audio_or_LLM'], 
                                model = gpt, layer = config.S_SETTINGS['FEATURE_LAYER'], context_words = config.GPT_WORDS, #if LMFeatures
                                stories = stories,  model_name = config.S_SETTINGS['S_MODEL'] #if audio model #Warning: get_audio_stim function argument should be model_name, not model!
                                )     
    features_2 = get_stim_features(config.S_SETTINGS_2['audio_or_LLM'], 
                                    model = gpt_2, layer = config.S_SETTINGS_2['FEATURE_LAYER'], context_words = config.GPT_WORDS, 
                                    stories = stories,  model_name = config.S_SETTINGS_2['S_MODEL'] 
                                    ) if args.stim_setting_2 is not None else None
    
    rstim, _, _ = get_stim(args.stim_setting, config.S_SETTINGS, stories, features, stacked = stacked ) #check if same  : np.vstack([rstim[story] for story in stories)                                 
    rstim_2, _, _ = get_stim(args.stim_setting_2, config.S_SETTINGS_2, stories, features_2, stacked = stacked ) if args.stim_setting_2 is not None else (None, None, None)
    
    return rstim, rstim_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--stim_setting", type = str, default = "default")
    parser.add_argument("--stim_setting_2", type = str, default = None) #! added for second stimulus setting (for encoding model) (only works if they are perfectly time matched (which is what we're modeling))
    parser.add_argument("--enc_setting", type = str, default = "default")
    parser.add_argument("--resp_setting", type = str, default = "default", 
                        choices = ["default",
                                   "default_32","default_128","default_256","default_512","default_1024", "default_2048", "default_4096", "default_8192", "default_16384", "default_32768",
                                   "PCA_10", "PCA_32","PCA_128", "PCA_256","PCA_1024", "PCA_512","PCA_2048", "PCA_4096","PCA_8192","PCA_16384","PCA_32768",
                                   "PCAonIMP_10", "PCAonIMP_32","PCAonIMP_128", "PCAonIMP_512", "PCAonIMP_1024", "PCAonIMP_2048", "PCAonIMP_4096","PCAonIMP_8192"])
    parser.add_argument("--name", type = str, default = "default_name") #used for misc_args!! (ex : CV or Bootstrap, etc..)
    args = parser.parse_args()

    config.initialize(args.stim_setting, args.stim_setting_2, args.enc_setting, args.resp_setting, args.name, args.subject) 
    
    # if args.enc_setting.startswith('singleRNN') : 
    #     import pdb ; pdb.set_trace()
    
    # training stories
    stories = eval(f"config.{args.subject}_train_list")
    
    #getting rstim 
    if args.stim_setting == "default":
        if args.enc_setting == "default":
            rstim, rstim_2 = legacy_when_default_stim(args, stories, stacked = True)
        else : 
            rstim, rstim_2 = legacy_when_default_stim(args, stories, stacked = False)
    else :  #use pretrained stim features!
        rstim, _ , _ = load_stim_features(args.stim_setting, config.S_SETTINGS, stories, 
                                          config.S_SETTINGS['FEATURE_LAYER'], tr_stats = None, stacked = False, downsampled = True)
        rstim_2, _ , _ = load_stim_features(args.stim_setting_2, config.S_SETTINGS_2, stories, 
                                          config.S_SETTINGS['FEATURE_LAYER'], tr_stats = None, stacked = False, downsampled = True) if args.stim_setting_2 is not None else (None, None, None)
        
        #if enc_setting is default, then stack the rstim (for encoding model)
        if args.enc_setting == "default": 
            rstim = np.vstack([rstim[story] for story in stories])
            rstim_2 = np.vstack([rstim_2[story] for story in stories]) if args.stim_setting_2 is not None else None
    print("getting rstim done!")
    if args.enc_setting != "default":
        response = get_resp(args.resp_setting, args.subject, stories, stack = False, **config.R_SETTINGS)
        
        joblib_path_name_generator = lambda x : os.path.join(config.DUMP_DIR, f"{x}_{config.task}_{config.format_name_name}_{config.SUBJECT}.jbl")
        rstim_path, response_path, rstim_2_path = joblib_path_name_generator("rstim"), joblib_path_name_generator("response"), joblib_path_name_generator("rstim_2")
        if not os.path.exists(rstim_path) or not os.path.exists(response_path) or not os.path.exists(rstim_2_path) :
            print("dumping rstim, response, rstim_2")
            joblib.dump(rstim, rstim_path) ; joblib.dump(response, response_path) ; joblib.dump(rstim_2, rstim_2_path)
        del rstim, response, rstim_2
        
        DNN_main.setup_options() #setup things         
        study = DNN_main.run_optuna(args.subject, joblib.load(rstim_path), joblib.load(response_path), joblib.load(rstim_2_path))
        
        # For computing voxelwise things, why load here (late)? To save memory (when running multiple processes, some can fail here without issue)
        response_full = get_resp("default", args.subject, stories, stack = False) if args.resp_setting != "default" else joblib.load(response_path)
        
        DNN_main.retrain_and_save_best_trial(study, args.subject, joblib.load(rstim_path), joblib.load(response_path), joblib.load(rstim_2_path), "encoding_model_%s" % args.gpt, response_full)
        exit()
    
    # estimate encoding model
    # rstim, tr_stats, word_stats = get_stim(args.stim_setting, config.S_SETTINGS, stories, features)                                 #run gpt on stimulus (gpt embedding, do resampling) 
    rresp = get_resp(args.resp_setting, args.subject, stories, stack = True, **config.R_SETTINGS)
    
    if rstim_2 is not None : #! if multimodal, just stack!
        rstim = np.hstack([rstim, rstim_2])
    
    if args.stim_setting == "default":
        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))                              #i.e. divide the number of time points by 5 and then by chunklen #TODO : find, why 5??? ==> cuz validation : 1/5of samples 
        weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = True, alphas = config.ALPHAS, # use_corr: was True, not False!
            nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
        bscorrs = bscorrs.mean(2).max(0)
    else :  #! i.e. doing as in scaling law paper
        from ridge_utils import ridge as ridge_2
        config.ALPHAS = np.logspace(1, 4, 15)
        config.NBOOTS = 3
        config.chunklen = 20
        nchunks = int(len(rresp) * 0.25 / config.chunklen)
        weights, _ , alphas, bscorrs, _ = ridge_2.bootstrap_ridge(rstim, rresp, rstim, rresp, config.ALPHAS, config.NBOOTS, config.chunklen, nchunks, use_corr = True, single_alpha = False)
        bscorrs = bscorrs.mean(2).max(0)

    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
    
    
    # save
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), weights = weights, alphas = alphas, voxels = vox, stories = stories)