##model for training and testing the MNIST dataset by making either DNN or BNN given the width and depth

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import os 

import wandb
import optuna
import argparse
import datetime
import re
from torchinfo import summary
import logging
import sys

#so that importing from .. is possible.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

##custom modules from past #! TODO : change later (probably needed)
import utils_DNN
import train_model
from model import MLP, activations, temporal_transformer   
from data import get_dataset, get_dataloader, TrainValSplitIterator, ConcatDataset, CombineDataset
from model import multimodal_transformer_regression, multimodal_RNN_regression

##custom modules, added now 
import config 
from types import SimpleNamespace

#################일단은 여기서 다 돌아가게 한후, 그 후에 functionalize해서 밑에 넣자!!################
#TODO 2. : CV, bootstrap, etc. implement해서 실제로 완성시키기
#TODO 4. : put in voting option for CV, bootstrap should vote by default! 
#TODO 7. : option to modify whether train/val split based on stories or data points themselves.  ==> 이건 사실 기존 TrainValIndicesIterator불러와서 story list를 대로 넣어주면 될듯

#TODO FOR TODAY :
#TODO 6. : config.py에서 encoder initializ할때, fold, split by story등등이 되도록!

#TODO 0. : stitch it up with train_EM_for_main.py, and start doing it!

#! PCA embedding을 예측하게 하면, important component 랑 아닌 component랑 똑같이 중요도를 가지고 하게됨!! 이러면 안될것같음... reverse PCA한결과에서 loss를 계산하게 하던지 weighted (w.r.t. the eigen value) MSE 계산을 하던지 해야할듯 
    #!=> 일단 이것은 추후의 TODO. 일단은 8192개만 잘 예측하게 해도 충분히 meaningful할듯 
    
##########################################################################

#objects wandb needs to log, and hyperparameters, name and so on => need to be defined in a config file? (argparse?)? 

#####SOME CONSTANTS########
#*key should be config.E_SETTINGS['model'] (i.e. MLP, MEyeMLP, etc.)
#*value should be a list of [model class itself, list of arguments needed for that model (which will be pulled from config.E_SETTINGS)]
#input_size, output_size : 는 이미 주어질 것!!
MODEL_ARG_DICT = {
    "MLP" : [MLP.MLP, ['activation',"width", "batchnorm"]], #'batch_size',
    "MLPTanh" : [MLP.MLP, ['activation',"width", "batchnorm"]], # #! now do MultimodalMLP! 
    "MultimodalMLP" : [MLP.MultimodalMLP, ["hiddens", "activation", "batchnorm", "separating_layer"]], 
    "MultimodalMLPTanh" : [MLP.MultimodalMLP, ["hiddens", "activation", "batchnorm", "separating_layer"]], 
    "reducedLinear" : [MLP.MLP, ['activation',"width", "batchnorm"]], 
    "MEyeMLP" : [MLP.MEyeMLP, [None]], #add more!! also not use this yet!
    "dsfwTFresponly" : [temporal_transformer.single_pred_TF_decoder, ['num_layers', 'nhead', 'ff_hidden_size', 'rstim_window_size','resp_window_size'] ],
    "singleTFVecPred" : [multimodal_transformer_regression.MultimodalTFVecPred, ['head_type','token_selection_strategy','d_model','num_layers', 'nhead', 'ff_hidden_size', 'linear_first', 'masks', 'delays', 'pe_functions', 'modality_embedding','separate_modalities'] ],
    "singleRNNVecPred" : [multimodal_RNN_regression.MultimodalRNNVecPred, ['num_layers', 'model_type', 'bidirectional', 'linear_first', 'head_type', 'token_selection_strategy', 'delays', 'activation'] ],
    # "intercausalTFclspred" : [multimodal_transformer_regression.MultimodalTFVecPred, ['head_type','token_selection_strategy','d_model','num_layers', 'nhead', 'ff_hidden_size', 'linear_first', 'masks', 'delays', 'pe_functions', 'modality_embedding','separate_modalities'] ],
    #TODO : add from here!! add model option and so on! (dsfwTFresponly)
    #TODO : MORE IMPORTNAT : HYPER_HYPER_PARAM_LIST width parameter needs updateing.... not general enough!(ex : for transformer, it should be d_model or sth...)
    #TODO : just load 'em through **kwargs, 
    #TODO :주의 : 이렇게하면 study_name이 이상해짐!! ==> study name actually doesn't matter, so just set it as default
    #TODO : in fact, I don't even know of HYPER_HYPER_PARAM_LIST is needed atall! just remove it?>?
    #TODO : check with format_name_name and so on!! (TOMMROROW!!)
}

##########################################



def main(trial, options, dataset, dataset_2, do_optuna = True, dataset_full = None, dataset_full_2 = None) : 
    #TODO : add wandb stuff here
    #TODO : add logging stuff here (i.e. save model, save model summary, save model params, save model par
    ###########OPTUNA SUGGESTION#########
    suggested_values = train_model.get_optuna_suggestion(trial, options)

    #################CV or BOOTSTRAP LOOP START#######################
    ##TODO : ad-hoc implementaiton, later should be configurable with config.py
    split_by_story = False #TODO : CHANGE THIS LATER!!
    if split_by_story : #split by stories
        stories = dataset.stories #found by default
        indices_iterator =  TrainValSplitIterator(stories, method = "default",iterator_seed=0)
        #TODO : make sure indices iterator gives the same indices always!!
        #! TODO : 그냥 split_by_story False일때랑 같이, 그냥 dataset의 특정 indices를 불러오도록 하기! (i.e. minimal code chnage!! just do like below!)
        raise NotImplementedError("split by story not implemented yet, as the only dataset class currently made stacks the stories")
    else :  #split by data points
        indices_iterator =  TrainValSplitIterator(indices = np.arange(len(dataset)), method = config.E_SETTINGS['validation_method'], 
                                                  k = config.E_SETTINGS['k_folds'], iterator_seed=config.E_SETTINGS['iterator_seed'])
        
    #! remove later!! just here to make sure I don't do overfitting my self and tell myself that something isn't implemented yet!
    if config.E_SETTINGS['validation_method'] != "default_no_shuffle"  :
        assert config.E_SETTINGS['model'] in ["default", "MLP", "MLPTanh", "MultimodalMLP", "MultimodalMLPTanh", "reducedLinear", "singleTFVecPred", "singleRNNVecPred"] , "CV and other validation_method not implemented for non-default encoding setting"
    #!주의 : because temporal data, cross validaiton not easy!! also may need to do stuff like just testing the last 20% or sth!
    return_dict = {} if not do_optuna else None #dict to store what should  be returned by main, only used when not doing optuna
    
    fold_best_value_list = []
    for FOLD, (train_indices, valid_indices) in enumerate(indices_iterator) : #this approach works both for CV and bootstrap and default!!
        #train_indices can either be indices (list of integers) or stories (list of story names)
        print(f"====================FOLD : {FOLD}====================")
        print(f"train/validation number of indices : {len(train_indices)},{len(valid_indices)}  ")
        print(f"train_indices : {train_indices} | valid_indices : {valid_indices}")
        
        #################DATASET/DATALOADER#######################
        train_set = torch.utils.data.Subset(dataset, train_indices)
        valid_set = torch.utils.data.Subset(dataset, valid_indices)
        dataset_full_set = torch.utils.data.Subset(dataset_full, valid_indices) if dataset_full is not None else None
        dataset_full_2_set = torch.utils.data.Subset(dataset_full_2, valid_indices) if dataset_full_2 is not None else None
                
        if dataset_2 is not None :
            assert config.E_SETTINGS['combined_dataset_generator'] #i.e. must be defined in the config file
            train_set_2 = torch.utils.data.Subset(dataset_2, train_indices)
            valid_set_2 = torch.utils.data.Subset(dataset_2, valid_indices)
            #! TODO : put in subset of dataset_full on only the avlidation set thing
            
            final_train_set = config.E_SETTINGS['combined_dataset_generator'](train_set, train_set_2) 
            final_valid_set = config.E_SETTINGS['combined_dataset_generator'](valid_set, valid_set_2)
            final_full_set = config.E_SETTINGS['combined_dataset_generator'](dataset_full_set, dataset_full_2_set) if dataset_full is not None else None
            
            #! TODO : rstim2이쪽파트를 최대한 제거하고, train_set_2, valid_set_2등만 남기고 ==> combined dataset (final_dataest) ==> train_loader 이런식으로 flow를 하기!
            #! 즉, 여러 dataloader만드는게 아닌, dataset을 합쳐서 하나로 만드는 것!
            #! 단, concat아닐때는 make the __getitem__ of the dataset to return a dict of X and Y (or X1, X2, Y) or sth like that! (so that it can be used in the model)
            #! TODO : 
            # (done) 1. final_dataset, final_dataloader들로 변수들 rename,  => done 
            # (done) 2. ConcatDataset같이, dict로 합치는 dataset combination dataet짜기  => done (combined_dataset_generator)
            # (later) 3. multimodal MLP 만들어서 이 dataloader을 받아서 처리하도록 하기! => need to be done... later..(CombineDataset있으니 그것을 써서 하면될듯... 일단은 validation부터 하자)
            # (doing) 4. training시, foldwise validation score for each voxel (correlation or loss or sth... return하기! (need to use get_vox_pred or sth ))
        else : 
            train_set_2 = valid_set_2 = None     
            final_train_set, final_valid_set = train_set, valid_set   
            final_full_set = dataset_full_set
        #! 밑의 것은 그냥... if 문 나오게 해도 될듯 
        final_train_loader = get_dataloader(config.E_SETTINGS['dataloader_type'], final_train_set, options.batch_size, shuffle=True)
        final_valid_loader = get_dataloader(config.E_SETTINGS['dataloader_type'], final_valid_set, options.batch_size, shuffle=False)
        final_valid_loader_full = get_dataloader(config.E_SETTINGS['dataloader_type'], final_full_set, options.batch_size, shuffle=False) if dataset_full is not None else None #! no need to do dataset_2 dataset_full and so on cuz we're only using the response thing 
        #! TODO : put in the response full thing here
        
        #####################################################

        ########################MODEL#######################
        # extra_kwargs = {} #use later if needed
        net_class, arg_list = MODEL_ARG_DICT[options.model]
        if options.model in ['MLP', 'dsfwTFresponly', "MLPTanh", "reducedLinear", "singleTFVecPred", "singleRNNVecPred"] :  #single input
            input_size = final_train_set[0][0].shape[-1] #i.e. first item of dataset's first item (X or Y)'s shape's last value  
            if options.modify_output_dim :
                assert options.resp_setting == "default" #only default response setting can be used for this
                output_size = options.recon_n_comp 
            else : 
                output_size = final_train_set[0][1].shape[-1] 
                
            if config.encoding_setting.startswith("multi") : #! multimodal transforemr용으로! (deubgging용)
                import pdb ; pdb.set_trace()
                
            #TF : input_size_list라는 것을 받기때문에, (instead of input_size), 조금 따로 처리해야함
            if options.model in ['singleTFVecPred'] :
                input_sizes = [input_size] #! ad-hoc, but works for now (single modality).. 일단은 model type에는 맞음!
                net = net_class(input_sizes = input_sizes, output_size = output_size,  dropout = suggested_values['dropout'], #MODEL_ARG_DICT에서 미리 설정 못한 것들
                                **{arg : config.E_SETTINGS[arg] for arg in arg_list})
            elif options.model in ['singleRNNVecPred'] :                
                #if nli (no linear input embedding), then d_input = input_size
                d_input = input_size if options.d_input == "none" else options.d_input 
                
                #if nlo (no linear output head), then d_hidden = output_size (/2 if bidirectional)
                if options.d_hidden == "none" and options.bidirectional :
                    d_hidden = output_size//2
                elif options.d_hidden == "none" and not options.bidirectional : 
                    d_hidden = output_size
                else : 
                    d_hidden = options.d_hidden
                input_sizes = [input_size] 
                
                # import pdb ; pdb.set_trace()
                net = net_class(input_sizes = input_sizes, d_input = d_input, d_hidden = d_hidden, output_size = output_size,  dropout = suggested_values['dropout'], #MODEL_ARG_DICT에서 미리 설정 못한 것들
                                **{arg : config.E_SETTINGS[arg] for arg in arg_list})
                
                
                #TODO : for RNN regression : #'input_sizes', 'd_input', 'd_hidden', 'output_size', 'dropout',  #'norm_type', 
                
                
            else :  #MLP, MLPtanh, reducedLinear
                net = net_class(input_size = input_size, output_size = output_size,  dropout = suggested_values['dropout'], #MODEL_ARG_DICT에서 미리 설정 못한 것들
                                **{arg : config.E_SETTINGS[arg] for arg in arg_list})
            
            #reverting back to default response 
            net = options.recon_func(net)  if options.modify_output_dim else net
            #! also, implement validation voxel selection thing (retrain 파트 바꾸기 
            #! implement audio + LLM embedding 
        else : #multimodal input
            #TODO: FIX THIS!! 
            input_sizes = [train_set[0][0].shape[-1], train_set_2[0][0].shape[-1]] #! : AD HOC!! must change (mutlimodal하려고 했던것)
            #!: modify : make it more flexible!! maybe make make train_set like a list of train_sets or sth!!  (DICT??)
            #! make multimodal, where each modality can do multiple list!! ==> think about this for detail!!( concat? add to dict?)
            #TODO : make sure that rstim_window_size are used somehow!! (maybe during the model creation??)
            output_size = final_train_set[0][1].shape[-1] 
            net = net_class(input_sizes = input_sizes, output_size = output_size,  dropout = suggested_values['dropout'], #MODEL_ARG_DICT에서 미리 설정 못한 것들
                            **{arg : config.E_SETTINGS[arg] for arg in arg_list})
                            # **{arg : config.E_SETTINGS[arg] for arg in arg_list},
                            # **{arg : config.E_SETTINGS_2[arg] for arg in arg_list})
        
        #####################################################


        ######################OPTIMIZER######################
        if options.optimizer in ["SGD"] : 
            #SGD : momentum is used
            optimizer = train_model.get_optimizer(options.optimizer, params=net.parameters(), lr=suggested_values['lr'], weight_decay=suggested_values['weight_decay'], momentum=suggested_values["momentum"]) 
        else : 
            optimizer = train_model.get_optimizer(options.optimizer, params=net.parameters(), lr=suggested_values['lr'], weight_decay=suggested_values['weight_decay'])
        #####################################################

        #####################TRAINING######################
        train_loss_history, val_loss_history, best_value, best_model_state_dict = train_model.train(trial, net, optimizer=optimizer, criterion=options.loss_criteria, train_loader=final_train_loader, val_loader=final_valid_loader, 
                                                                                device="cuda", num_epochs=options.epochs, early_stopping_metric= options.early_stopping_metric,
                                                                                early_stopping_patience= options.early_stopping_patience)

        fold_best_value_list.append(best_value)
        ####################################################
        
        #####################optional saving if not doing optuna######################
        if not do_optuna :
            target_recon_voxelwise_corr, target_recon_voxelwise_mse, target_recon_voxelwise_mae, true_voxelwise_corr, true_voxelwise_mse, true_voxelwise_mae = train_model.evaluate_model_voxel_wise(net, final_valid_loader, final_valid_loader_full)
            return_dict[f'target_recon_voxelwise_corr_{FOLD}'] = target_recon_voxelwise_corr
            return_dict[f'target_recon_voxelwise_mse_{FOLD}'] = target_recon_voxelwise_mse
            return_dict[f'target_recon_voxelwise_mae_{FOLD}'] = target_recon_voxelwise_mae
            return_dict[f'true_voxelwise_corr_{FOLD}'] = true_voxelwise_corr
            return_dict[f'true_voxelwise_mse_{FOLD}'] = true_voxelwise_mse
            return_dict[f'true_voxelwise_mae_{FOLD}'] = true_voxelwise_mae
                        
            return_dict[f'model_state_dict_{FOLD}'] = best_model_state_dict
            return_dict[f'optimizer_state_dict_{FOLD}'] = optimizer.state_dict()
            return_dict[f'train_loss_history_{FOLD}'] = train_loss_history
            return_dict[f'val_loss_history_{FOLD}'] = val_loss_history
            return_dict[f'best_value_{FOLD}'] = best_value
            return_dict[f'train_indices_{FOLD}'] = train_indices #! TEST LATER IF IT WORKS
            return_dict[f'val_indices_{FOLD}'] = valid_indices

            #! TODO : reload the best_model_state_dict, run for loop on CV things... then on the outputs ..... sth sth 
            #! correlation map을 return하기? or r^2 값 for each fold return? => this seems more reasonable since ensemble model은 validation이 없고, eahc fold-wise model에 validation이 있으니
            #! 따라서, fold-wise r2값을 구하자 => 아니다. 전체 concat해서 해도될듯?
        ###############################################################################
    #################MODEL SUMMARY#######################
    data, target = next(iter(final_valid_loader)) #one data from valid loader 
    #review net architecture
    print(net)
    if type(data) == list :
        simple_model_stats, verbose_model_stats, model_param_count = "None", "None", 0 #cuz multimodalMLP 같은  것들은 data가 list라서 조금 복잡해짐!
    else : 
        verbose_model_stats, simple_model_stats, model_param_count = utils_DNN.get_model_summary(net, data.shape)
        print("total_num_params : ", model_param_count)

    #TODO : save verbose_model_stats, simple_model_stats, model_param_count to wandb or sth
    #####################################################
    
    #################saving other things too if not doing optuna (i.e. during "retrain_and_save_best_trial"#######################
    if not do_optuna :
        return_dict["verbose_model_stats"] = str(verbose_model_stats)
        return_dict["simple_model_stats"] = str(simple_model_stats)
        return_dict["model_param_count"] = model_param_count
        return_dict['avg_fold_best_values'] = np.mean(fold_best_value_list)    
        #model_arch_infos : for later use, to know what model was used (for evaulate_encoder.py)
        if options.model == "singleTFVecPred" :
            return_dict['model_arch_infos'] = {'model' : options.model, 'input_sizes' :  input_sizes , 'output_size' :  output_size,
                                               **{arg : config.E_SETTINGS[arg] for arg in arg_list}}
        elif options.model == "singleRNNVecPred" :
            return_dict['model_arch_infos'] = {'model' : options.model, 'input_sizes' : input_sizes, 'd_input' :  d_input , 'd_hidden' :  d_hidden, 'output_size' :  output_size,
                                               **{arg : config.E_SETTINGS[arg] for arg in arg_list}}
        elif options.model.startswith("MultimodalMLP") :
            return_dict['model_arch_infos'] = {'model' : options.model, 'input_sizes' :  input_sizes , 'output_size' :  output_size,
                                               **{arg : config.E_SETTINGS[arg] for arg in arg_list}}
        else : 
            return_dict['model_arch_infos'] = {'model' : options.model, 'input_size' :  input_size , 'output_size' :  output_size,
                                            **{arg : config.E_SETTINGS[arg] for arg in arg_list}} 
        
        if config.encoding_setting.startswith("multi") : #! multimodal transforemr용으로! (deubgging용)
                import pdb ; pdb.set_trace()
                print("만약 multi이면 위에서 return_dict['model_arch_infos']에 들어가는게 달라져야함! (single이랑은 또 다를 수 있어서 일부러 남겨둠))")
                
        #! TODO : change this so that input_size is input_sizes if single pred thing 
    
    # store auxiliary infos (train,test loss and acc, param_count) on the trial (for easier access later)
    trial.set_user_attr("last_train_loss", train_loss_history[-1])
    trial.set_user_attr("last_val_loss", val_loss_history[-1])
    trial.set_user_attr("model_param_count", model_param_count) 
        
    if do_optuna :
        return np.mean(fold_best_value_list) #return the average of the best values from each fold
    else : 
        return return_dict

def setup_options() :
    """ 
    #for setting up options and others.. necessary to be called before doing run_optuna or sth 
    """
    global options
    ####################config and options#################################    
    options = SimpleNamespace(**config.E_SETTINGS, 
                            **{"stim_setting" : config.stimulus_setting, "stim_setting_2" : config.stimulus_setting_2, "enc_setting" : config.encoding_setting, 
                                "resp_setting" : config.response_setting, "name" : config.name_setting, "task" : config.task, "format_name_name" : config.format_name_name})
    if options.do_optuna :
        assert len(options.lr) == len(options.momentum) == len(options.weight_decay) == len(options.dropout)== 2, "lr, momentum, weight_decay must have the same length" 
    #######################################################################

####main######
def run_optuna(subject, train_X_dict, train_Y_dict, train_X_dict_2 = None):  
    """ 
    runs optuna on "subject" using the folowing data.
        
    train_X_dict : stimulus dictinoary (key : story)
    train_Y_dict : response dictionary (key : story) ##SAME Y for both X and X_2!! (duh)
    train_X_dict_2 : second stimulus (modality) dictionary (key : story) (optional)    
    """
    ####################DATA#################################    
    curr_dir = os.path.dirname(__file__)

    dataset = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict, train_Y_dict) #TODO : "plain" 보다 더 확장시키기!! make this into option, later to be changed depending on enc_model and so on 
    dataset_2 = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict_2, train_Y_dict) if train_X_dict_2 is not None else None
    del train_X_dict, train_Y_dict  # Delete the objects to save memory
    if train_X_dict_2 is not None: del train_X_dict_2  # Conditional deletion
    #####################################################
    
    #optuna.db dirs and stuff
    ckpt_dir = os.path.dirname(__file__) #current dir을 다시 가져오기 
    db_dir = f"{ckpt_dir}/optuna"
    os.makedirs(ckpt_dir, exist_ok=True)
    url = "sqlite:///" + os.path.join(db_dir, f"{subject}_{options.task}_{options.format_name_name}_optuna.db")
    # Add stream handler of stdout to show the messages 
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = optuna.storages.RDBStorage( url = url, heartbeat_interval = 60, grace_period = 120)

    n_trials = options.n_trials
    
    study = optuna.create_study(study_name = "default",
                                    storage = storage,
                                    load_if_exists = True,
                                    direction = 'minimize',
                                    pruner = optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                        n_warmup_steps=10, #5
                                                                        interval_steps=1), patience = 10))  #5
                                    #TODO removed pruner... may add later
    #TODO add enqueue_trial later? (or just do GridSampler?)
    #study.enqueue_trial({"learning_rate" : args.lr_0, "weight_decay" : args.wd_0}, skip_if_exists = True)
    
    #####exit the thing if already more than n_trials#####
    cancel = False
    try : 
        if (study.trials_dataframe()['state'] == "COMPLETE").sum() >= n_trials : 
            print(f"already more than {n_trials} trials done!")
            return study #return without doing optuna 
    except :
        pass
    ######################################################
    # print("n_trials : ", n_trials)
    # print("number of completed trials : ", (study.trials_dataframe()['state'] == "COMPLETE").sum())
    print("don't go further than here!!")
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_trials//4)
    study.optimize(lambda trial : main(trial, options, dataset, dataset_2, True), 
                   callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,))])
                   #https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    return study 

    #TODO : make save location to the config.MODEL_DIR or sth!
    #TODO : implement validation method bootstrapping (from config.E_SETTINGS['validation_method'])
    #TODO : make sure iterator always gives the same indices (for reproducibility)
    #TODO : make sure the get_dataset can do other thnigs than "plain" 

def retrain_and_save_best_trial(study, subject, train_X_dict, train_Y_dict, train_X_dict_2 = None, save_name = None, train_Y_dict_full = None): 
    """ 
    retrains the best model (i.e. retrain with the best hyperparam from optuna) from the study and saves the results as save_name.pth (on MODEL_DIR/subject/)
    """
    #study should be ran at least run n_trial times 
    assert (study.trials_dataframe()['state'] == "COMPLETE").sum() >= options.n_trials, "study should be ran at least n_trial times"
    best_trial = study.best_trial
    
    #!below line : MUST BE CHANGED, in accordance with get_datset above!!!
    dataset = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict, train_Y_dict) #TODO : "plain" 보다 더 확장시키기!! make this into option, later to be changed depending on enc_model and so on
    dataset_2 = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict_2, train_Y_dict) if train_X_dict_2 is not None else None
    dataset_full = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict, train_Y_dict_full) if train_Y_dict_full is not None else None 
    dataset_full_2 = get_dataset(config.E_SETTINGS['dataset_type'], train_X_dict_2, train_Y_dict_full) if ((train_Y_dict_full is not None) and (train_X_dict_2 is not None)) else None 
    #! dataset_full should probably be changed if doing RNN 
    print("주의 : 주의 : 주의 : 주의 : !! below line : MUST BE CHANGED, in accordance with get_datset above!!!")
    #########################################################
    
    retrain_results = main(best_trial, options, dataset, dataset_2, False, dataset_full, dataset_full_2) #train the best model again (without optuna. works since best_trial is frozen trial)
    
    #assert less than 5% error from optuna validation value and retrained validation value 
    assert (np.abs(retrain_results['avg_fold_best_values'] - best_trial.value))/best_trial.value < 0.05, f"retrain value : {retrain_results['avg_fold_best_values']} | optuna value : {best_trial.value}"
    
    #save the things retrain_results dict
    save_location = os.path.join(config.MODEL_DIR, subject)
    os.makedirs(save_location, exist_ok = True)
    torch.save(retrain_results, os.path.join(save_location, f"{save_name}.pth"))

"""
BELOW : not used, so 생략됨.. but maybe could be used later
1. db_dir을 ./ 가 아닌 config.MODEL_DIR에 하기 ==> 안한 이유 : 비교가 오히려 어려워질듯 
# db_dir = os.path.join(config.MODEL_DIR, config.subject)
# os.makedirs(db_dir, exist_ok=True) #don't do optuna folder.. unnecessary
# url = "sqlite:///" + os.path.join(db_dir, f"{options.task}_optuna.db") ##attempts at svaing the thing at MODEL_DIR, but nah... too complicated
"""
