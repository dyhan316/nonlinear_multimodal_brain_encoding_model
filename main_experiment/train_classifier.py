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

from DNN import DNN_main
import joblib
np.random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    
    parser.add_argument("--stim_setting", type = str, default = "default")
    parser.add_argument("--stim_setting_2", type = str, default = None) 
    
    
    parser.add_argument("--enc_setting", type = str, default = "default")
    parser.add_argument("--resp_setting", type = str, default = "default", 
                        choices = ["default",
                                   "default_32","default_128","default_256","default_512","default_1024", "default_2048", "default_4096", "default_8192", "default_16384", "default_32768",
                                   "PCA_10", "PCA_32","PCA_128", "PCA_256","PCA_1024", "PCA_512","PCA_2048", "PCA_4096","PCA_8192","PCA_16384","PCA_32768",
                                   "PCAonIMP_10", "PCAonIMP_32","PCAonIMP_128", "PCAonIMP_512", "PCAonIMP_1024", "PCAonIMP_2048", "PCAonIMP_4096","PCAonIMP_8192"])
    parser.add_argument("--name", type = str, default = "default_name") #used for misc_args!! (ex : CV or Bootstrap, etc..)
    args = parser.parse_args()


    quit() #! for now, just quit here! (for debugging)
    
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
    # TODO: Make the following work according to enc_model! 
    # TODO!! CHANGE THIS!! (temporarily removed for later use) DO LATER LATER (temporarily moved to != "default" case, but should be reworked!)
    if args.enc_setting != "default":
        response = get_resp(args.resp_setting, args.subject, stories, stack = False, **config.R_SETTINGS)
        
        if args.enc_setting.startswith("multi") :  #! multimodal TF, not implemented yet!, for debugging!
            import pdb ; pdb.set_trace()
        
        joblib_path_name_generator = lambda x : os.path.join(config.DUMP_DIR, f"{x}_{config.task}_{config.format_name_name}_{config.SUBJECT}.jbl")
        rstim_path, response_path, rstim_2_path = joblib_path_name_generator("rstim"), joblib_path_name_generator("response"), joblib_path_name_generator("rstim_2")
        if not os.path.exists(rstim_path) or not os.path.exists(response_path) or not os.path.exists(rstim_2_path) :
            print("dumping rstim, response, rstim_2")
            joblib.dump(rstim, rstim_path) ; joblib.dump(response, response_path) ; joblib.dump(rstim_2, rstim_2_path)
        del rstim, response, rstim_2
        
        DNN_main.setup_options() #setup things         
        study = DNN_main.run_optuna(args.subject, joblib.load(rstim_path), joblib.load(response_path), joblib.load(rstim_2_path))
        
        # for computing voxelwise things, why load here (late)? : to save memory (some processes can die here when running multiple simultaneously)
        response_full = get_resp("default", args.subject, stories, stack = False) if args.resp_setting != "default" else joblib.load(response_path)
        
        DNN_main.retrain_and_save_best_trial(study, args.subject, joblib.load(rstim_path), joblib.load(response_path), joblib.load(rstim_2_path), "encoding_model_%s" % args.gpt, response_full)
        # actually encoding_model_percieved.pth should include not only the above but also the estimate noise model below! ==> should we save everything together at the bottom?
        raise NotImplementedError("noise model not implemented... so stop here!!! temporarily kept for testing!") 
    
    # estimate encoding model
    # rstim, tr_stats, word_stats = get_stim(args.stim_setting, config.S_SETTINGS, stories, features)                                 #run gpt on stimulus (gpt embedding, do resampling) 
    rresp = get_resp(args.resp_setting, args.subject, stories, stack = True, **config.R_SETTINGS)
    
    if rstim_2 is not None : #! if multimodal, just stack!
        rstim = np.hstack([rstim, rstim_2])
    
    #! in scaling law : int(len(Rresp) * 0.25 / chunklen) was used
    if args.stim_setting == "default":
        nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))                              #i.e. divide the number of time points by 5 and then by chunklen #TODO : find, why 5??? ==> cuz validation : 1/5of samples 
        weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = True, alphas = config.ALPHAS, # use_corr: was set to True not False!
            nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
        bscorrs = bscorrs.mean(2).max(0)
    else :  # i.e. if scaling law paper, follow the ipynb procedure!
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