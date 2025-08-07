import numpy as np
import os
import json
import config
from utils_resp import get_PCA_mat, transform_data, inverse_transform_data
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from DNN.DNN_main import MODEL_ARG_DICT
from types import SimpleNamespace
import copy

from DNN.data import create_inout_sequences, batch_second_collate_fn, get_dataset

##! BELOW : SKIP FOR NOW... just doing ad hoc implementation!
# def get_torch_inp_from_input_data(enc_setting, input_data, **encoding_model_dict) :
#     """
#     from the input data, get the torch input, with batch things too!! (i.e. convert to torch tensor)
#     """
#     if enc_setting.startswith("MLP_"):
#         torch_inp = torch.from_numpy(input_data).float().to(config.EM_DEVICE)
#     elif enc_setting.startswith("RNN_"):
#         raise NotImplementedError("not implemented yet")
#     elif enc_setting.startswith("dsfwTF_"):
#         raise NotImplementedError("not implemented yet")
#     elif enc_setting.startswith("dsfwTFresponly_"):
#         X, X_shifted = create_inout_sequences(input_data, encoding_model_dict['model_arch_infos']['resp_window_size'], 1)
#         raise NotImplementedError("not implemented yet")
#     else : 
#         raise NotImplementedError("not implemented yet")
#     return torch_inp    

def get_model_pred(enc_setting, input_data, validation_method, output_data = None, **encoding_model_dict) : 
    """
    from the input data and things, gets model output s
    input_data 
    
    (주의 : this is irrespective of the resp setting (i.e. PCA 를 키든말든 상관안함..(just do model prediction))
    """
    if "default" == enc_setting: #linear regression
        weights = encoding_model_dict['weights']
        model_pred = input_data@weights
        model_pred_list = [model_pred]  #TODO : if default enc_setting should also take in CV or sth, modify!!
        ensemble_pred_dict = None 
    else : #DNN 
        encoding_model_dict_copy = copy.deepcopy(encoding_model_dict) #don't want .pop("model") to affect the original dict
        #getting model architecutre
        model_arch_infos = encoding_model_dict_copy['model_arch_infos']
        options_model = model_arch_infos.pop("model")
        #loading model dicts
        model_fold_key_list  = [key for key in encoding_model_dict_copy.keys() if key.startswith('model_state_dict_')]
        
        ##torch_inp만들기 (i.e. convert input_data to be used for model) (MLP, 등등에 따라 다르기때문에 if 로 만듬)
        if  enc_setting.startswith("MLP") or enc_setting.startswith("reducedLinear") :
            torch_inp = torch.from_numpy(input_data).float().to(config.EM_DEVICE) #TODO : may need to change for different inputs!! (ex : RNN or Transformer) (don't use the dataset or anythnig... just use input data only)
        elif enc_setting.startswith("MultimodalMLP") : 
            torch_inp = [torch.from_numpy(data_i).float().to(config.EM_DEVICE) for data_i in input_data]
        elif config.E_SETTINGS['dataset_type'] == "down_sample_fixed_window_vec_pred" and enc_setting.startswith("single"): 
            #! single 을 조건으로 넣어준 이유 : multkmodal 쓰면 다르게 해야할수도 있어서
            X_dict, Y_dict = {"test_story" : input_data}, {"test_story" : output_data}
            tf_dataset = get_dataset(config.E_SETTINGS['dataset_type'], X_dict, Y_dict)
            torch_inp = tf_dataset.stacked_X_input.permute(1,0,2).float() #bc model is fp32
            torch_output = tf_dataset.stacked_Y_output.float() #not used            
        else : 
            raise NotImplementedError("not implemented yet")
        
        model_pred_list = []
        for model_fold_key in model_fold_key_list:
            with torch.no_grad():
                print(model_fold_key)
                net_class, arg_list = MODEL_ARG_DICT[options_model]
                #* outputdim을 맞추는 것 (when reocnMSE and such) 은 필요없다 (이미 model_arch_infos에 다 들어가있음)
                net = net_class(**model_arch_infos)
                net = config.E_SETTINGS['recon_func'](net) if config.E_SETTINGS['modify_output_dim'] else net
                if config.E_SETTINGS['dataset_type'] == "down_sample_fixed_window_vec_pred" : 
                    if model_arch_infos['token_selection_strategy'] in ["concat", "weighted_pool"] :
                        #adhoc, initialize the model by passing the input once
                        net(torch_inp.cpu()) #initialize the thing                
                net.load_state_dict(encoding_model_dict_copy[model_fold_key])
                net.eval()
                net.to(config.EM_DEVICE)
                model_pred = net(torch_inp) #may need to add bach dimension
                model_pred_list.append(model_pred)    
    
        ##ensemble predictions    
        if validation_method == "CV":
            ensemble_pred_dict = {
            "mean" : ensemble_prediction(config, model_pred_list, 'mean', weights = None ),
            "median" : ensemble_prediction(config, model_pred_list, 'median', weights = None ),
            "weighted" : None #TODO : implement this later if time permits!
            } #weighted = ensemble_prediction(config, model_pred_list, 'weighted', weights = ?? )
            ##TODO : weighted voting 하기!!        
        else : 
            raise NotImplementedError("not implemented yet")

    return model_pred_list, ensemble_pred_dict

#TODO : validation method에 따라서 다르게 해야함.. also, weighted AUROC 라던지 하려면 weighting 값도 받거나, encoding_model_dict에서 가져와야함!
#!#!#!
#also, option to use wieghted voting or not !! (actually, bound to the vlaidaiotn method!!) 
# #* : 위의 경우, validation_method를 참조하기! 
    
def get_vox_pred_from_model_pred(resp_setting, model_pred, subject, **resp_setting_dict) : 
    """
    from the model prediction, get the voxel prediction (kinda like reverse resp_setting)
    """
    if "default" == resp_setting:
        model_vox_pred = model_pred #no need to do anything 
    elif resp_setting.startswith("PCA_"):
        model_vox_pred = inverse_transform_data(model_pred, subject, 
                                                n_comp = int(resp_setting.split("_")[1]))
    else : 
        raise NotImplementedError("not implemented yet")    
        
    return model_vox_pred

def ensemble_prediction(config, pred_list, stat_measure, weights = None ):
    """
    https://machinelearningmastery.com/voting-ensembles-with-python/
    https://jermwatt.github.io/machine_learning_refined/notes/11_Feature_learning/11_9_Bagging.html
    * stat_measure : 'mean', 'median', 'weighted', 'hard_mode', or 'hard_weighted'
    * weights = weights to use if weighted
    
    will only do soft voting!! 
    * mean
    * median
    * weighted
    """
    #converting to np.array in case it's list
    PRED_stack = torch.stack(pred_list) 
    PRED_stack = PRED_stack.cpu().numpy() 
    weights = np.array(weights) if weights is not None else None
    
    if stat_measure == "mean" : 
        PRED_summary = PRED_stack.mean(axis=0)
    elif stat_measure == "median" : 
        PRED_summary = np.median(PRED_stack, axis = 0)
    elif stat_measure == "weighted" :
        PRED_summary = (PRED_stack.T @ weights)/weights.sum()            
    else : 
        raise ValueError(f"{stat_measure} is not one of the stat measures that can be used")
    return PRED_summary
