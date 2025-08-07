import torch.optim as optim
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import optuna
import copy

from utils_DNN import PCA_weighted_MSE
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# from utils_enc_model import get_vox_pred_from_model_pred
from utils_eval import compute_r_single_trial, np_mse_loss, np_mae_loss

import config

#optimizer options
OPTIMIZER_DICT = {
    "SGD" : optim.SGD,
    # "Adam" : optim.Adam,
    # "RMSprop" : optim.RMSprop,
    "Adagrad" : optim.Adagrad,
    # "Adadelta" : optim.Adadelta,
    "AdamW" : optim.AdamW,
    # "SparseAdam" : optim.SparseAdam,
    # "Adamax" : optim.Adamax,
    # "ASGD" : optim.ASGD,
    "LBFGS" : optim.LBFGS,
    # "Rprop" : optim.Rprop,
}


def get_optimizer(optimizer_name, **kwargs):
    assert optimizer_name in OPTIMIZER_DICT
    return OPTIMIZER_DICT[optimizer_name](**kwargs) 

def early_stopping(metric_history, patience=5, direction="maximize"):
    r""" 
    if the metric has not improved in the last patience epochs, stop training    
    and return the best metric value along with whether to early stop or not 
    """
    metric_history = np.array(metric_history)
    if direction == "maximize":
        best_idx = np.argmax(metric_history)
    elif direction == "minimize":
        best_idx = np.argmin(metric_history)
    else:
        raise ValueError("direction must be either 'maximize' or 'minimize'")

    if len(metric_history) - 1 - best_idx >= patience:
        return metric_history[best_idx], True
    else:
        return metric_history[best_idx], False    
    

#TODO : implement AMP, DDP if time permits

#Training Loop, with tqdm, given the model, optimizer, loss function, and data
def train(trial, net, optimizer, criterion, train_loader, val_loader, 
          device, num_epochs, early_stopping_metric, early_stopping_patience):
    net.to(device)
    net.eval()
    initial_val_loss = 0
    with torch.no_grad():
        # for data, target in val_loader:
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = send_to_device(data, target, device)
            output = net(data)
            loss = criterion(output, target)
            initial_val_loss += loss.item()
        print(f"Initial Val Loss: {initial_val_loss/len(val_loader)}")
    metric_history = []
    train_loss_history, val_loss_history = [], []
        
    for epoch in range(num_epochs):
        #utils.print_mean_std_shape(net) #TODO log this with wandb later!
        net.train()
        train_loss = 0
        train_correct = 0
        val_loss = 0
        val_correct = 0
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = send_to_device(data, target, device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        # Calculate validation accuracy
        net.eval()
        with torch.no_grad():
            # for data, target in val_loader:
            for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
                data, target = send_to_device(data, target, device)
                output = net(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        train_loss_history.append(train_loss) ; val_loss_history.append(val_loss)
        
        ############reporting intermediate values and logging in optuan and wandb
        print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        if trial is not None:
            trial.report(val_loss, epoch) 
            # if trial.should_prune(): ##TODO TODO : UNCOMMENT THIS!! #TODO : turned off pruning!
            #     raise optuna.TrialPruned()
            
        #TODO do wandb logging
        ########################################################
        
        
        ################early stopping##########################
        if early_stopping_metric == "val_loss":
            metric_history.append(val_loss)
            early_stop_direction = "minimize"
        else : 
            raise NotImplementedError(f"early_stopping_metric {early_stopping_metric} is not implemented yet")
        # elif early_stopping_metric == "val_acc":
        #     metric_history.append(val_acc)
        #     early_stop_direction = "maximize"
        
        best_value, early_stop = early_stopping(metric_history, 
                                                patience=early_stopping_patience,
                                                direction=early_stop_direction)
        if metric_history[-1] == best_value:
            best_model_state_dict = copy.deepcopy(net.state_dict())
            
        #print how long along early stopping 
        if early_stop:
            print(f"Early stopping at epoch {epoch+1}, the best value is {best_value}")
            break
        else : 
            print(f"Early stopping patience {len(metric_history) - 1 - np.argmin(metric_history)} out of {early_stopping_patience}")
        ########################################################
    return train_loss_history, val_loss_history, best_value, best_model_state_dict

def get_optuna_suggestion(trial, options):
    # Create a dictionary to store the suggested values
    suggested_values = {}

    # Discrete uniform
    if options.model == "reducedLinear":
        dropout = 0.0
    else : 
        dropout = trial.suggest_float("dropout", options.dropout[0], options.dropout[1], step=0.1)
    # Continuous log
    lr = trial.suggest_float("lr", options.lr[0], options.lr[1], log=True)
    weight_decay = trial.suggest_float("weight_decay", options.weight_decay[0], options.weight_decay[1], log=True)

    ########things that might not be optimized depending on things ##########
    if options.optimizer in ["SGD"] : 
        momentum = trial.suggest_float("momentum", options.momentum[0], options.momentum[1], step=0.1)  
        suggested_values["momentum"] = momentum
    else :
        pass
    #########################################################################
    
    # Add the suggested values to the dictionary
    suggested_values["dropout"] = dropout
    suggested_values["lr"] = lr
    suggested_values["weight_decay"] = weight_decay

    return suggested_values

#! may have to do the saving of validation on defalut 
def evaluate_model_voxel_wise(net, valid_loader, valid_loader_full): 
    from utils_enc_model import get_vox_pred_from_model_pred #! import here to avoid circular import
    r""" 
    * Evaluate the model voxel-wise, and return the voxel-wise MSE and the correlation
    
    net : the net
    valid_loader : the loader for the validation set
    valid_loader_full : the loader for the full voxerl values (for calculating voxel-wise things)
    
    net(valid_loader) then convert to voxel, then compare with valid_loader_full targets
    
    """
    aggregate_output, aggregate_target = inference_on_model(net, valid_loader)
    _, voxelwise_target = inference_on_model(net, valid_loader_full, only_target=True)
    
    aggregate_output_vox_pred = get_vox_pred_from_model_pred(config.response_setting, aggregate_output, config.SUBJECT)
    aggregate_target_vox_pred = get_vox_pred_from_model_pred(config.response_setting, aggregate_target, config.SUBJECT)
    
    #slew of metrics (MSE, correlation, etc)
    target_recon_voxelwise_corr = compute_r_single_trial(aggregate_output_vox_pred, aggregate_target_vox_pred)
    target_recon_voxelwise_mse = np_mse_loss(np.array(aggregate_output_vox_pred), np.array(aggregate_target_vox_pred),axis=0)
    target_recon_voxelwise_mae = np_mae_loss(np.array(aggregate_output_vox_pred), np.array(aggregate_target_vox_pred),axis=0)

    true_voxelwise_corr = compute_r_single_trial(aggregate_output_vox_pred, voxelwise_target)
    true_voxelwise_mse = np_mse_loss(np.array(aggregate_output_vox_pred), np.array(voxelwise_target),axis=0)
    true_voxelwise_mae = np_mae_loss(np.array(aggregate_output_vox_pred), np.array(voxelwise_target),axis=0)
    
    return  target_recon_voxelwise_corr, target_recon_voxelwise_mse, target_recon_voxelwise_mae, true_voxelwise_corr, true_voxelwise_mse, true_voxelwise_mae
    
    
def inference_on_model(net, val_loader, only_target=False):
    net.eval()
    output_tnsr_list = []
    target_tnsr_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = send_to_device(data, target, config.EM_DEVICE)
            output = net(data)
            output_tnsr_list.append(output.cpu()) if not only_target else None
            target_tnsr_list.append(target.cpu())
            
    #-2 so that even if using RNN, the shape is (T, N, D) then -2 is N, the batch dimension
    if only_target:
        return None, torch.cat(target_tnsr_list, dim=-2)
    else : 
        return torch.cat(output_tnsr_list, dim=-2), torch.cat(target_tnsr_list, dim=-2) 
    
    
def send_to_device(data, target, device):
    if isinstance(data, list):
        data = [tnsr.to(device) for tnsr in data]
    else:
        data = data.to(device)
    target = target.to(device)
    return data, target