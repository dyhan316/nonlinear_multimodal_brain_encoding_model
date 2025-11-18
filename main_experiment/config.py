import os
import numpy as np
from torch import nn
import shlex
# from DNN.utils_DNN import PCA_weighted_MSE, PCA_recon_after_model #moved to avoid circular import 
from DNN.model import utils_transformer
from DNN.data import ConcatDataset, CombineDataset
from functools import partial
import re
import warnings
from typing import Dict, Any

# paths
# Set your data directory path here
REPO_DIR = "/scratch/connectome/ahhyun724/NeuroBabyLM/data_and_weights"
# Alternative path configuration:
# REPO_DIR = "/alternative/path/to/semantic_decoding_data"
DATA_LM_DIR = os.path.join(REPO_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data_train")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models") 
RESULT_DIR = os.path.join(REPO_DIR, "results")
SCORE_DIR = os.path.join(REPO_DIR, "scores")
DUMP_DIR =  os.path.join(REPO_DIR, "dump") #'/tmp' is faster... but unless doing non-ram dataset, this is not that much different... #scratch is TOO slow... using /tmp is better #where we will store temporary npy and joblib and soon... (to save memory)
STIM_FEATURE_EXTRACTED_DIR = os.path.join(REPO_DIR, "extracted_stim_features") #MUST BE DONE BEFORE IMPORTING TRANSFORMERS to take effect


SCALING_LAW_DATA_DIR = os.path.join(REPO_DIR, 'scaling_law_data')

DATA_AUDIO_DIR = os.path.join(SCALING_LAW_DATA_DIR, "story_audio") #! added!! #! change later to accomodate all audio files!

#story list that is used!
S1_train_list = ['itsabox', 'odetostepfather', 'inamoment',  'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
S2_train_list = ['itsabox', 'odetostepfather', 'inamoment', 'afearstrippedbare', 'findingmyownrescuer', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'escapingfromadirediagnosis', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'marryamanwholoveshismother', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'thesecrettomarriage', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
S3_train_list = S2_train_list.copy()
test_list = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']

#save huggingface models in scratch
os.makedirs(f"{REPO_DIR}/HF_HOME", exist_ok=True)
os.environ['HF_HOME'] = os.path.join(REPO_DIR, "HF_HOME") #f"{REPO_DIR}/HF_HOME"

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10) #! in paper, np.logspace(1, 4, 15) was used 
NBOOTS = 50 #! in encoding model scaling law, 3 was used!
# NBOOTS = 2 #testing purpose. original : 50 takes too long! #! change later when actually running!
VOXELS = 10000
# VOXELS = 100 #testing purpose #! CHANGE LATER WHEN ACTUALLY RUNNING!
CHUNKLEN = 40 #! in scalign laws, 20 was used
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters 

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# model that can be used 
LLM_MODEL_LIST = ['GPT1', 'GPT2', 'LLAMA7B', 'LLAMA13B', 'LLAMA30B', 'LLAMA65B', 
 'LLAMA27Bhf', "LLAMA213Bhf", "LLAMA270Bhf" ,'LLAMA38B', 'LLAMA370B', 'ds_no_front_trim', 'babylm-babyllama-10m-2024'
 'babylm-ltgbert-10m-2024']
AUDIO_MODEL_LIST = ["whisper-tiny", "whisper-small", "whisper-base", "whisper-large", 
                    "whisper-large-v2", "whisper-large-v3",  #! same num params but newer.. try this?
                    "wavlm-base", "wavlm-base-plus", "wavlm-large",
                    "hubert-base", "hubert-large", "hubert-xlarge"]

NUM_LAYERS_DICT = {
    'GPT1' : 12,
    'GPT2' : 12,
    'LLAMA7B' : 32 -1, #do LLMs later 
    'LLAMA27Bhf' : 32 -1 , #do LLMs later 
    'LLAMA38B' : 32 - 1, #do LLMs later 
    'LLAMA13B' : 40 - 1, #do LLMs later 
    'LLAMA30B' : 60 - 1, #do LLMs later 
    'LLAMA65B' : 80 - 1, #do LLMs later 
    'babylm-babyllama-10m-2024' : 16,
    'babylm-ltgbert-10m-2024': 12,
    # audio model num_layers: +1 is added! (because of the embedding layer I think)
    'whisper-tiny' : 5,
    'whisper-small' : 13,
    'whisper-base' : 7,
    'whisper-large' : 33,
    'whisper-large-v2' : 33,
    'whisper-large-v3' : 33,
    'wavlm-base' : 13,
    'wavlm-base-plus' : 13,
    'wavlm-large' : 25 ,
    'hubert-base' : 13,  
    'hubert-large' : 25,
    'hubert-xlarge' : 49, 
}

# devices

GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"

S_SETTINGS = {} #stim setting, initialize as nothing
S_SETTINGS_2 = {} #stim setting 2, initialize as nothing
E_SETTINGS = {} #enc setting
R_SETTINGS = {} #resp setting

#initializing if not default, (ex : doing PCA, LLAMA, ... )
def initialize(stim_setting, stim_setting_2, enc_setting, resp_setting, name, subject): 
    from DNN.utils_DNN import PCA_weighted_MSE, PCA_recon_after_model #moved to avoid circular import
    global stimulus_setting, stimulus_setting_2,  encoding_setting, response_setting, name_setting, SUBJECT
    stimulus_setting = stim_setting
    stimulus_setting_2 = stim_setting_2
    encoding_setting = enc_setting
    response_setting = resp_setting
    name_setting = name
    SUBJECT = subject
    
    global MODEL_DIR, RESULT_DIR, SCORE_DIR, task, format_name_name
    
    #modify model names
    task = f"{stim_setting}-{enc_setting}-{resp_setting}" if stim_setting_2 is None else f"{stim_setting}__{stim_setting_2}-{enc_setting}-{resp_setting}"
    format_name_name = format_name(name)
    mid_folder = f"exp_results/{task}/{format_name_name}" 
    MODEL_DIR = os.path.join(REPO_DIR, mid_folder, "models")  
    RESULT_DIR = os.path.join(REPO_DIR, mid_folder, "results")
    SCORE_DIR = os.path.join(REPO_DIR, mid_folder, "scores")
    
    global S_SETTINGS #global stim setting cuz stim_initialize could be reused. 
    
    stim_initialize(stim_setting, S_SETTINGS, name)
    if stim_setting_2 is not None : 
        global S_SETTINGS_2
        stim_initialize(stim_setting_2, S_SETTINGS_2, name)
    resp_initialize(resp_setting, name)
    enc_initialize(enc_setting, name)
    misc_initialize(name)

def misc_initialize(misc_args_str):
    """ 
    for handling misc options separately (ex : CV or Bootstrap or sth..)
    """ 
    global S_SETTINGS, E_SETTINGS, R_SETTINGS, S_SETTINGS_2
    
    #i.e. skip this part if not needed (ex : when runninf default)
    if encoding_setting == "default":
        assert misc_args_str == "default" or misc_args_str == "default_name"
    
    if misc_args_str == "default" or misc_args_str == "default_name":
        if encoding_setting != "default":
            raise NotImplementedError("if enc is not default (i.e. DNN), then misc args (ex : validation method and loss_criteria, and so on) should be given!")
        ###! IMP : must set default values (None)
        E_SETTINGS['validation_method'] = None
        E_SETTINGS['loss_criteria'] = None
        E_SETTINGS['multimodal_method'] = None 
        return None     
    
    misc_args_list = shlex.split(misc_args_str)
    misc_args_dict = {misc_args_list[i]: misc_args_list[i + 1] for i in range(0, len(misc_args_list), 2)}
    
    #custom configuration when not default 
    if "validation_method" in misc_args_dict.keys():
        if misc_args_dict['validation_method'] == "CV":
            E_SETTINGS['validation_method'] = "CV"
            E_SETTINGS['iterator_seed'] = 0
            E_SETTINGS['k_folds'] = 5 #i.e. 5 fold CV 
        elif misc_args_dict['validation_method'] == "bootstrap":
            E_SETTINGS['validation_method'] = "bootstrap"
            E_SETTINGS['iterator_seed'] = 0
            E_SETTINGS['k_folds'] = 50 #i.e. 50 bootstrapping 
            raise NotImplementedError("bootstrap not implemented yet")
        elif misc_args_dict['validation_method'] == "default_no_shuffle":
            E_SETTINGS['validation_method'] = "default_no_shuffle"
            E_SETTINGS['iterator_seed'] = None
            E_SETTINGS['k_folds'] = 5 #i.e. 1/5 = 20% is  test set
        elif misc_args_dict['validation_method'] == "default_shuffle":
            E_SETTINGS['validation_method'] = "default"
            E_SETTINGS['iterator_seed'] = 0
        else : 
            raise NotImplementedError(f"validation method {misc_args_dict['validation_method']} not implemented yet")
    else : 
        raise ValueError("validation_method should be given!")
        
    E_SETTINGS['modify_output_dim'] = False #default
    if "loss_criteria" in misc_args_dict.keys():
        if misc_args_dict['loss_criteria'] == "MSE":
            E_SETTINGS['loss_criteria'] = nn.MSELoss()
        elif misc_args_dict['loss_criteria'] == "MAE":
            E_SETTINGS['loss_criteria'] = nn.L1Loss()
        elif misc_args_dict['loss_criteria'].startswith("PCAreconMSE_"):
            E_SETTINGS['modify_output_dim'] = True
            E_SETTINGS['recon_n_comp'] = int(misc_args_dict['loss_criteria'].split("_")[1])
            R_SETTINGS["PCA_MODEL"] = os.path.join(REPO_DIR, "new_exp_files/PCA_weights") 
            E_SETTINGS['recon_func'] = partial(PCA_recon_after_model, SUBJECT)
            E_SETTINGS['loss_criteria'] = nn.MSELoss() 
            E_SETTINGS['early_stopping_patience'] = 5
            print("early stopping patience changed to 5 since doign recon loss!")
        elif misc_args_dict['loss_criteria'].startswith("PCAreconMAE_"):
            E_SETTINGS['modify_output_dim'] = True
            E_SETTINGS['recon_n_comp'] = int(misc_args_dict['loss_criteria'].split("_")[1])
            R_SETTINGS["PCA_MODEL"] = os.path.join(REPO_DIR, "new_exp_files/PCA_weights") 
            E_SETTINGS['recon_func'] = partial(PCA_recon_after_model, SUBJECT)
            E_SETTINGS['loss_criteria'] = nn.L1Loss()
            E_SETTINGS['early_stopping_patience'] = 5
            print("early stopping patience changed to 5 since doign recon loss!")
        elif misc_args_dict['loss_criteria'] == "PCA_weighted_MSE":
            E_SETTINGS['loss_criteria'] = PCA_weighted_MSE(SUBJECT)
        else : 
            raise ValueError(f"loss_criteria {misc_args_dict['loss_criteria']} not implemented yet")
    else : 
        raise ValueError("loss_criteria should be given!")

    if "multimodal_method" in misc_args_dict.keys():
        assert S_SETTINGS_2 #must have S_SETTINGS_2 if multimodal_method is given
        
        multimodal_method = misc_args_dict['multimodal_method']
        if multimodal_method == "concat_dataset": #concat dataset featurewise
            E_SETTINGS['combined_dataset_generator'] = ConcatDataset
            E_SETTINGS['input_combiner'] = lambda x, y : np.concatenate((x, y), axis = -1) #!ad-hoc solution.. will not work if using transformer
        elif multimodal_method.startswith("multimodal_MLP"):
            # encoder setting overlaps with MLP... where should this be placed?
            E_SETTINGS['combined_dataset_generator'] = CombineDataset
        else :
            raise ValueError(f"multimodal_method {multimodal_method} not implemented yet")
        # need to add stuff like when using transformer, concat dataset might be different, etc.
    
            
def stim_initialize(stim_setting, setting, name):
    global LLM_MODEL_LIST, AUDIO_MODEL_LIST
    ###################===LANGUAGE STIMS (LLMs)===##################
    #setting here would be S_SETTINGS or S_SETTINGS_2
    if stim_setting == "default":
        setting['S_MODEL'] = "GPT1"
        setting['FEATURE_LAYER'] = GPT_LAYER #i.e. default : 9
    elif stim_setting.startswith("GPT2_") : 
        setting['S_MODEL'] = "GPT2"
        setting['FEATURE_LAYER'] = int(stim_setting.split('_')[1])
        setting['LLM_CHECKPOINT'] = "openai-community/gpt2"
        setting['LLM_TOKENIZER'] = "openai-community/gpt2"
    elif stim_setting.startswith("LLAMA") :
        s_model, feature_layer, hf_model_name = parse_general_settings(stim_setting)
        assert s_model in LLM_MODEL_LIST
        setting['S_MODEL'] = s_model
        setting['FEATURE_LAYER'] = feature_layer
        setting['HF_MODEL_NAME'] = hf_model_name
    elif stim_setting.startswith("huggyllama") : #!TODO : can be merged (huggyllama, babylm, EleutherAI)
        s_model, feature_layer, hf_model_name = parse_general_settings(stim_setting)
        #assert s_model in LLM_MODEL_LIST
        setting['S_MODEL'] = s_model
        setting['FEATURE_LAYER'] = feature_layer
        setting['HF_MODEL_NAME'] = hf_model_name
    elif stim_setting.startswith("babylm") : # changed from ("babylm-babyllama-10m-2024") after feature extraction
        s_model, feature_layer, hf_model_name = parse_general_settings(stim_setting)
        #assert s_model in LLM_MODEL_LIST
        setting['S_MODEL'] = s_model
        setting['FEATURE_LAYER'] = feature_layer
        setting['HF_MODEL_NAME'] = hf_model_name
    elif stim_setting.startswith("EleutherAI") :
        s_model, feature_layer, hf_model_name = parse_general_settings(stim_setting)
        #assert s_model in LLM_MODEL_LIST
        setting['S_MODEL'] = s_model
        setting['FEATURE_LAYER'] = feature_layer
        setting['HF_MODEL_NAME'] = hf_model_name
    ###################===AUDIO STIMS===##################
    ##audio models #! may need to add more audio options?
    elif any(stim_setting.startswith(model) for model in AUDIO_MODEL_LIST): 
        setting['audio'] = True
        setting.update(parse_audio_options(stim_setting)) 
        # TODO: FEATURE_LAYER here is a list not a single number... may need to be changed later? (shouldn't matter for pre-extracted features though)
    
    ###################===MISC STIMS===##################    
    #! TODO : only leave option for DSNFT 
    elif stim_setting.startswith('DSNFT') :  
        llm_or_audio_model_to_use = '_'.join(stim_setting.split('_')[1:])
        if llm_or_audio_model_to_use.startswith("LLAMA") :
            s_model, feature_layer, hf_model_name = parse_llama_settings(llm_or_audio_model_to_use)
            assert s_model in LLM_MODEL_LIST
            setting['S_MODEL'] = s_model
            setting['FEATURE_LAYER'] = feature_layer
            setting['HF_MODEL_NAME'] = hf_model_name
        elif any(llm_or_audio_model_to_use.startswith(model) for model in AUDIO_MODEL_LIST): 
            setting['audio'] = True
            setting.update(parse_audio_options(llm_or_audio_model_to_use)) 
        else : 
            print("DEFAULT will be used! (GPT1, 9th layer, and so on)")    
            setting['S_MODEL'] = "GPT1" #! should be changed!! 
            setting['FEATURE_LAYER'] = GPT_LAYER #i.e. default : 9
        # below : FIXED!! shouldn't be changed
        setting['TRIM_STIM_LAST_WORDS']  = 5 #remove the last 5 words (cuz response fMRI data does not exist for that, and future value of stim is not needed)
        setting['sync_len'] = 10 # number of excess words at the beginning with no corresponding response
    else : 
        raise NotImplementedError(f"setting {stim_setting} not implemented yet")

    # indicate audio_or_LLM
    if setting['S_MODEL'] in LLM_MODEL_LIST:
        setting['audio_or_LLM'] = "LLM"
    elif setting['S_MODEL'] in AUDIO_MODEL_LIST:
        setting['audio_or_LLM'] = "audio"
    else : 
        setting['audio_or_LLM'] = "LLM"
        warnings.warn("HARCODED setting['audio_or_LLM'] as LLM")
        #raise ValueError(f"setting {stim_setting} not implemented yet... must specify whether audio or LLM")

def resp_initialize(resp_setting, name):
    global VOXELS, R_SETTINGS
    # assert resp_setting != "default" #if default, no need to initialize
    
    #default, but different number of voxels to take 
    if resp_setting.split("_")[0] == "default":
        pass 
    elif "default" in resp_setting.split("_")[0]: # default_XXX format 
        VOXELS = int(resp_setting.split("_")[1])
            
    #PCA settings
    elif "PCA" == resp_setting.split("_")[0]: 
        R_SETTINGS["PCA_N_COMP"] = int(resp_setting.split("_")[1])
        R_SETTINGS["PCA_MODEL"] = os.path.join(REPO_DIR, "new_exp_files/PCA_weights")
        VOXELS = R_SETTINGS["PCA_N_COMP"] #TODO : option to change later, for say,50%or sth..
    
    elif "PCAonIMP" == resp_setting.split("_")[0]:
        R_SETTINGS["PCA_N_COMP"] = int(resp_setting.split("_")[1])
        R_SETTINGS["PCA_MODEL"] = os.path.join(REPO_DIR, "new_exp_files/PCA_on_important_features_weights/resulting_PCAonIMP_weights")
        VOXELS = R_SETTINGS["PCA_N_COMP"]

def enc_initialize(enc_setting, name):
    global E_SETTINGS
    
    ##global settings for all DNNs (default)
    if enc_setting == "default":
        E_SETTINGS['save_type'] = "numpy"  #TODO : later for use in DNN, change to torch (pth instaed of numpy and so on)
        pass
    else :
        E_SETTINGS['do_optuna'] = True
        E_SETTINGS['save_type'] = "torch" #i.e. if enc_setting != default, will save in pth instead of npz 
        E_SETTINGS['seed'] = 2024
        
        #training settings 
        E_SETTINGS['optimizer'] = "AdamW"
        E_SETTINGS['epochs'] = 200 #! #TODO : CHANGE LATER (make custom for each enc_setting) (currenlty ,only 10 for testing purposes)
        E_SETTINGS['early_stopping_metric'] = "val_loss"
        E_SETTINGS['early_stopping_patience'] = 10
        if response_setting == "default": #! if default setting, reduce epoc h and early stopping patience 
            E_SETTINGS['epochs'] = 50
            E_SETTINGS['early_stopping_patience'] = 5
        # E_SETTINGS['loss_criteria'] = nn.MSELoss() #TODO : maybe add option to change later (ex : mse, mae, ...)
        
        #optuna settings #actually can be made into single float values, in which case just running 
        E_SETTINGS['dropout'] = [0.1, 0.3]
        E_SETTINGS['lr'] = [1e-5, 1e-1]
        E_SETTINGS['weight_decay'] = [5e-5, 1e-1]
        E_SETTINGS['momentum'] = [0.1, 0.5] #for SGD, which will not be used.. but still included just in case         
    
    ##specific settings for each DNN
    if enc_setting == "default" : 
        pass   
    elif enc_setting.startswith("MLP_") : 
        #dataset and dataloader configs #set up here and not stim-setting cuz dataset/datalader depends on what encoder I use 
        E_SETTINGS['dataset_type'] = "plain" 
        E_SETTINGS['dataloader_type'] = "plain"
        
        # general configs, applies to all types (this section below should be copied and modified for RNN as well)
        E_SETTINGS['n_trials'] = 70 #!shoudl be 100, 5 for fast prototyping (7 for testing since minimum is 5 for pruner)
        E_SETTINGS['model'] = "MLP"
        E_SETTINGS['batch_size'] = 128 #TODO : change later if needed!
        
        #MLP specific settings
        E_SETTINGS['width'] =  [int(j) for j in enc_setting.split('_')[1:]] #ex : MLP_200_200_200_200 ==> [200, 200, 200, 200]
        E_SETTINGS['activation'] = "relu"
        E_SETTINGS['batchnorm'] = "batchnorm" #or "layernorm" or False
    elif enc_setting.startswith("MLPTanh_") :
        #! same as MLP_ except activation is tanh 
        E_SETTINGS['dataset_type'] , E_SETTINGS['dataloader_type'] = "plain" , "plain"
        E_SETTINGS['n_trials'] = 70 ; E_SETTINGS['model'] = "MLPTanh" ; E_SETTINGS['batch_size'] = 128
        E_SETTINGS['width'] =  [int(j) for j in enc_setting.split('_')[1:]] ; E_SETTINGS['activation'] = "tanh"
        E_SETTINGS['batchnorm'] = "batchnorm"  
    elif enc_setting.startswith("reducedLinear_") :
        #batchnorm is false and activation is Identity, and no dropout (i.e. two linear matricies (one hidden layer))(~reduced rank Linear regression)
        E_SETTINGS['dataset_type'] , E_SETTINGS['dataloader_type'] = "plain" , "plain"
        E_SETTINGS['n_trials'] = 70 ; E_SETTINGS['model'] = "reducedLinear" ; E_SETTINGS['batch_size'] = 128
        E_SETTINGS['width'] =  [int(j) for j in enc_setting.split('_')[1:]] ; E_SETTINGS['activation'] = "identity"
        E_SETTINGS['batchnorm'] = "False"      
        assert len(E_SETTINGS['width']) == 1 #only one hidden layer
        #! TODO : in thing, input dropout should be 0! 
    elif enc_setting.startswith("MultimodalMLP_") : 
        #ex : MultimodalMLP_2,3,4,6_2 ==> [10,2,3,4,6]
        # E_SETTINGS['input_sizes'] = [int(j) for j in enc_setting.split('_')[1].split(',')] #ex : MultimodalMLP_4,8_10_2,3,4,6_2 ==> [4,8]
        dict_for_update = {
            'dataset_type' : "plain" , 'dataloader_type' : "plain",
            'n_trials' : 70 , 'model' : "MultimodalMLP" , 'batch_size' : 128,
            'width' : [int(j) for j in enc_setting.split('_')[1].split(',')] , 'separating_layer' : int(enc_setting.split('_')[2]),
            'activation' : "relu" , 'batchnorm' : "batchnorm", 'hiddens' : [int(j) for j in enc_setting.split('_')[3].split(',')]
        }
        E_SETTINGS.update(dict_for_update)       
    elif enc_setting.startswith("MultimodalMLPTanh_") : 
        #python train_EM_for_main.py --subject S1 --gpt perceived --stim_setting LLAMA7B_12 --stim_setting_2 whisper-large_final_default_default --enc_setting MultimodalMLP_16384,5120_1_256 --resp_setting PCA_512 --name "validation_method CV loss_criteria MAE multimodal_method multimodal_MLP"
        dict_for_update = {
            'dataset_type' : "plain" , 'dataloader_type' : "plain",
            'n_trials' : 70 , 'model' : "MultimodalMLP" , 'batch_size' : 128,
            'width' : [int(j) for j in enc_setting.split('_')[1].split(',')] , 'separating_layer' : int(enc_setting.split('_')[2]),
            'activation' : "tanh" , 'batchnorm' : "batchnorm", 'hiddens' : [int(j) for j in enc_setting.split('_')[3].split(',')]
        }
        E_SETTINGS.update(dict_for_update)              

            
    ####vector pred RNNs! (instead of predicting whole sequences)
    elif enc_setting.startswith("singleRNN-") : #* Single Modality, everything else customizable!
        E_SETTINGS['model'] = "singleRNNVecPred"
        add_single_RNN_settings(enc_setting)
    
    ####vector pred transformers! (instead of predicting whole sequences)
    #ex :  single-cosi-caus-avgp_1_64_1_8_128_10 : single modality, cosine position emedding, causal mask
    elif enc_setting.startswith("single-") : #* Single Modality, everything else customizable! 
        E_SETTINGS['model'] = "singleTFVecPred"
        add_transformer_settings(enc_setting)
        TF_other_settings_dict = make_generic_TF_settings_dict(num_modalities = 1 , modality_embedding = False, separate_modalities = False, delay = E_SETTINGS['delay'], 
                                                                        **parse_single_TF_str(enc_setting))
        E_SETTINGS.update(TF_other_settings_dict)

        #single-cosi-caus-avgp : single modality, cosine position emedding, causal mask, avgpooling 
        #modality options = sing : single, multi : multi 
        #PE options = cos : cosine, revc : reverse cosine, learn : learnt positional embedding
        #mask options = full : full mask, caus : causal mask
        #token selection stragety = clsp : cls token, last : last token, avgp : avg pooling, maxp : max pooling
        
        # (maxpool not implemented.. skipped)
        
    elif enc_setting.startswith("dsfwTFresponly_") : #response only (predict response)
        add_transformer_settings(enc_setting) #add trailing options to E_SETTINGS
        E_SETTINGS['dataset_type'] = "down_sample_fixed_window_resp_only"  
        E_SETTINGS['dataloader_type'] = "batch_second"
        
        # General configs, applies to all types (including RNN - these settings should be copied and adjusted appropriately)
        E_SETTINGS['n_trials'] = 70 # Should be changed later!! 5 is just for testing purposes!!
        E_SETTINGS['model'] = "dsfwTFresponly"
        E_SETTINGS['batch_size'] = 128
    
    else : 
        raise NotImplementedError(f"setting {enc_setting} not implemented yet")

def add_single_RNN_settings(enc_setting):
    global E_SETTINGS 
    model_map = {'reluRNN' : 'RNN','RNN': 'RNN', 'LSTM': 'LSTM', 'GRU': 'GRU'}
    input_proj_map = {'nli': False, 'li': True}
    output_proj_map = {'nlo': None, 'lo': 'linear'} #implement MLP head later if needed
    token_map = {'last': 'last', 'avgp': 'avg_pool', 'maxp': 'max_pool', 'conc' : 'concat', 'weip' : 'weighted_pool'}
    bidirectional_map = {'uni': False, 'bi': True}
    
    parts = enc_setting.split('_')
    _, model_type, linear_first_elem, head_type, bidirectional, token_selection_strategy = parts[0].split('-')
    
    E_SETTINGS['model_type'] = model_map[model_type]
    E_SETTINGS['linear_first'] = [input_proj_map[linear_first_elem]]
    E_SETTINGS['head_type'] = output_proj_map[head_type]
    E_SETTINGS['bidirectional'] = bidirectional_map[bidirectional]
    E_SETTINGS['token_selection_strategy'] = token_map[token_selection_strategy]
    
    delay, num_layers, rstim_window_size, d_input, d_hidden = [convert_to_int_if_possible(i) for i in parts[1:6]]
    if linear_first_elem == 'nli':
        assert d_input == "none" #cuz input : LLM size if nli 
    if head_type == 'nlo':
        assert d_hidden == "none" #cuz hidden : pca//2 if bi, pca if uni (i.e. predefined)    
    E_SETTINGS['delay'], E_SETTINGS['num_layers'], E_SETTINGS['rstim_window_size'], E_SETTINGS['d_input'], E_SETTINGS['d_hidden'] = delay, num_layers, rstim_window_size, d_input, d_hidden
    E_SETTINGS['delays'] = [delay] #(single modality)
        
    if model_type == 'reluRNN':
        E_SETTINGS['activation'] = 'relu'
    else : 
        E_SETTINGS['activation'] = 'tanh'
    E_SETTINGS.update(generic_RNN_settings_dict())

def convert_to_int_if_possible(string):
    try:
        return int(string)
    except :
        return string

def generic_RNN_settings_dict():
    return {
        'dataset_type': "down_sample_fixed_window_vec_pred",
        'dataloader_type': "batch_second",
        'n_trials': 70,
        'batch_size': 128
    }

def add_transformer_settings(enc_setting):
    global E_SETTINGS
    #! dsfwTFresponly : legacy,  will not be used
    if enc_setting.startswith("dsfwTFresponly_") :
        E_SETTINGS['num_layers'] = int(enc_setting.split('_')[1])
        E_SETTINGS['nhead'] = int(enc_setting.split('_')[2])
        E_SETTINGS['ff_hidden_size'] = int(enc_setting.split('_')[3])
        E_SETTINGS['rstim_window_size'] = int(enc_setting.split('_')[4])
        E_SETTINGS['resp_window_size'] = int(enc_setting.split('_')[5])
        print(f"num_layers: {E_SETTINGS['num_layers']}, nhead: {E_SETTINGS['nhead']}, ff_hidden_size: {E_SETTINGS['ff_hidden_size']}, rstim_window_size: {E_SETTINGS['rstim_window_size']}, resp_window_size: {E_SETTINGS['resp_window_size']}")
    else : 
        E_SETTINGS['delay'] = int(enc_setting.split('_')[1])
        E_SETTINGS['d_model'] = int(enc_setting.split('_')[2])
        E_SETTINGS['num_layers'] = int(enc_setting.split('_')[3])
        E_SETTINGS['nhead'] = int(enc_setting.split('_')[4])
        E_SETTINGS['ff_hidden_size'] = int(enc_setting.split('_')[5])
        E_SETTINGS['rstim_window_size'] = int(enc_setting.split('_')[6])
        
def parse_single_TF_str(enc_setting):
    # Example: single-cosi-caus-avgp_1_64_1_8_128_10 extracts: single modality, cosine position embedding, causal mask (to pass to `make_generic_TF_settings_dict`)
    # Define the mapping for each part of the setting
    modality_map = {'sing': 'single', 'multi': 'multi'}
    pe_map = {'cosi': utils_transformer.Cosine_PE, 'revc': utils_transformer.reverse_Cosine_PE, 'lear': utils_transformer.Learned_PE}
    mask_map = {'full': 'within_full', 'caus': 'within_causal'}
    token_map = {'clsp': 'cls_token', 'last': 'last', 'avgp': 'avg_pool', 'maxp': 'max_pool'}
    
    # Split the input string by underscores
    parts = enc_setting.split('_')
    
    # Split the first part by hyphens to get individual settings
    _, pe_option, mask_option, token_str_option = parts[0].split('-')
    
    # Create the dictionary with parsed values
    parsed_dict = {
        'positional_embedding': pe_map[pe_option],
        'mask_option': mask_map[mask_option],
        'token_selection_strategy': token_map[token_str_option]
    }
    return parsed_dict


def make_generic_TF_settings_dict(num_modalities: int, modality_embedding: bool, separate_modalities: bool, 
                                    token_selection_strategy: str, mask_option: str, delay: int, positional_embedding: Any) -> Dict[str, Any]:
    return {
        #dataset, dataloader settings 
        'dataset_type': "down_sample_fixed_window_vec_pred",
        'dataloader_type': "batch_second",
        
        #(multi)-modality related settings 
        'modality_embedding': modality_embedding,
        'separate_modalities': separate_modalities,
        
        #misc settings
        'head_type': 'linear',
        'linear_first': [True] * num_modalities, #! may need to be chagned to False if direct mapping... (for saving)
        'token_selection_strategy': token_selection_strategy,
        'masks': get_multimodal_mask_options(num_modalities, mask_option),
        'delays': [delay] * num_modalities,
        'pe_functions': [positional_embedding] * num_modalities,
        'n_trials': 70,
        'batch_size': 128
    }

def get_multimodal_mask_options(num_modalities, mask_option):
    if mask_option == "inter_causal":
        mask = [['causal'] * num_modalities for _ in range(num_modalities)]
    elif mask_option == "within_causal":
        mask = [['causal' if i == j else None for j in range(num_modalities)] for i in range(num_modalities)]
    elif mask_option == "inter_full":
        mask = [['full'] * num_modalities for _ in range(num_modalities)]
    elif mask_option == "within_full":
        mask = [['full' if i == j else None for j in range(num_modalities)] for i in range(num_modalities)]
    elif mask_option == "within_full_inter_causal" : 
        mask = [['full' if i == j else 'causal' for j in range(num_modalities)] for i in range(num_modalities)]
        
    else:
        raise ValueError(f"Invalid mask_option: {mask_option}")
    return mask

def commas2list(commas):
    return [int(i) for i in commas.split(",")]

def format_name(name):
    # Split the name into a list of arguments
    args = name.split()

    # Group the arguments in pairs and join them with double underscores
    formatted_args = ['__'.join(args[i:i+2]) for i in range(0, len(args), 2)]

    # Join the formatted arguments with a hyphen
    formatted_name = '-'.join(formatted_args)

    return formatted_name

def reverse_format_name(formatted_name):
    # Split the formatted name on hyphens to get the argument pairs
    formatted_args = formatted_name.split('-')

    # Split each argument pair on double underscores and flatten the list
    args = [arg for pair in formatted_args for arg in pair.split('__')]

    # Join the arguments with spaces to reconstruct the original name
    original_name = ' '.join(args)

    return original_name

def parse_general_settings(stim_setting):
    feature_layer = int(stim_setting.split('_')[-1])
    stim_prefix = stim_setting.rsplit('_', 1)[0]
    hf_model_name = stim_prefix.replace('/', '_', 1)
    return stim_prefix, feature_layer, hf_model_name
    if stim_setting.startswith("huggyllama") or stim_setting.startswith("meta-llama"):
        #! lucy changed below in TRAIN step. worked fine with parse_llama_settings for extract_features
        stim_prefix = stim_setting.split('_')[0]
        feature_layer = int(stim_setting.split('_')[1])
        hf_model_name = stim_prefix.replace('/', '_', 1)
        return stim_prefix, feature_layer, hf_model_name
        #return parse_llama_settings(stim_setting)
    else: #for other models
        #hf_model_name = stim_setting.replace('/', '-', 1)
        if stim_setting.count('_') > 2:
            feature_layer = int(stim_setting.split('_')[-1])
            stim_prefix = stim_setting.rsplit('_', 1)[0]
            hf_model_name = stim_prefix.replace('/', '_', 1)
            return stim_prefix, feature_layer, hf_model_name
        stim_prefix = stim_setting.split('_')[0]
        feature_layer = int(stim_setting.split('_')[1])
        hf_model_name = stim_prefix.replace('/', '_', 1)
        return stim_prefix, feature_layer, hf_model_name


def parse_llama_settings(stim_setting):
    # Mapping of LLAMA model prefixes to their Hugging Face model names
    llama_models = {
        "LLAMA7B": "huggyllama/llama-7b",
        "LLAMA13B": "huggyllama/llama-13b",
        "LLAMA30B": "huggyllama/llama-30b",
        "LLAMA65B": "huggyllama/llama-65b",
        "LLAMA27Bhf": "meta-llama/Llama-2-7b-hf",
        "LLAMA213Bhf": "meta-llama/Llama-2-13b-hf",
        "LLAMA270Bhf": "meta-llama/Llama-2-70b-hf",
        "LLAMA38B": "meta-llama/Meta-Llama-3-8B",
        "LLAMA370B": "meta-llama/Meta-Llama-3-70B",
        'babylm-babyllama-10m-2024' : 'babylm/babyllama-10m-2024',
        'babylm-ltgbert-10m-2024' : 'babylm/ltgbert-10m-2024'
    }
    for prefix, hf_model_name in llama_models.items():
        if stim_setting.startswith(prefix):
            feature_layer = int(stim_setting.split('_')[1])
            return prefix, feature_layer, hf_model_name
    return None, None, None

def parse_audio_options(input_str):
    # Default values (in milliseconds)
    default_contextsz = 16000
    default_chunksz = 100
    
    # Split the input string by '_'
    parts = input_str.split('_')
    
    audio_model = parts[0]
    
    # Extract layers
    num_layers = NUM_LAYERS_DICT[audio_model]
    layer_str = parts[1]
    if layer_str == "final":    
        layer = layer_str
    elif layer_str.isdigit():
        layer = int(layer_str)
    elif layer_str == "all":
        layer = list(range(num_layers))
    elif layer_str == "even":
        layer = list(range(1, num_layers, 2))
        print(layer)
    elif layer_str == "odd":
        layer = list(range(0, num_layers, 2))
    elif len(layer_str.split(',')) > 1:
        layer = [int(l) for l in layer_str.split(',')]
    else : 
        raise ValueError(f"layer {layer_str} not implemented yet")
    
    # Extract contextsz and chunksz, use defaults if necessary
    contextsz = default_contextsz if parts[2] == "default" else int(parts[2]) # * 1000 in ms!
    chunksz = default_chunksz if parts[3] == "default" else int(parts[3])
    return {"S_MODEL" : audio_model, "FEATURE_LAYER": layer, "contextsz": contextsz, "chunksz": chunksz}


# def format_folder_name(name):
#     # Split the name into a list of arguments
#     args = name.split('--')

#     # Remove leading and trailing spaces from each argument and replace spaces with underscores
#     formatted_args = [re.sub(' ', '_', arg.strip()) for arg in args if arg]

#     # Join the formatted arguments with a hyphen
#     formatted_name = '-'.join(formatted_args)

#     return formatted_name
