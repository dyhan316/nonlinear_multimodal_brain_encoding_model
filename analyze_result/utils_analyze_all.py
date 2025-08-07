import joblib 
import numpy as np 
import os 
import sys 
sys.path.append('../main_experiment')
# import config 
# from utils_resp import get_test_resp
import matplotlib.pyplot as plt

import numpy as np

def get_error_stats_over_time(all_test_resp, *model_vox_preds, error_type='mse', normalize=True):
    for model_vox_pred in model_vox_preds:
        assert all_test_resp.shape == model_vox_pred.shape, "Shapes of inputs do not match. If doing tensessions, should run this ten times"
        
    if error_type == 'mse':
        error_func = lambda x, y: np.abs(x - y) ** 2
    elif error_type == 'mae':
        error_func = lambda x, y: np.abs(x - y)
    elif error_type == 'zero_dummy': #for testing purposes
        error_func = lambda x, y: np.zeros_like(x)
    else:
        raise NotImplementedError("Only 'mse' and 'mae' are supported for error_type.")    
    
    ##Initializing arrays for discrete and continuous metrics
    num_time_points, num_voxels = all_test_resp.shape
    model_better_over_time, model_worst_over_time = np.empty((num_time_points, num_voxels), dtype='object'), np.empty((num_time_points, num_voxels), dtype='object') #discrete
    relative_error_difference_over_time, error_difference_over_time = np.empty((num_time_points, num_voxels)), np.empty((num_time_points, num_voxels)) #continuous
    model_errors_over_time = [np.empty((num_time_points, num_voxels)) for _ in model_vox_preds] #continuous
    
    for time_point in range(num_time_points):
        errors = [error_func(all_test_resp[time_point, :], model_vox_pred[time_point, :]) for model_vox_pred in model_vox_preds]
        model_better_per_vox, model_worst_per_vox = np.argmin(errors, axis=0), np.argmax(errors, axis=0)
        model_better_over_time[time_point, :], model_worst_over_time[time_point, : ] = model_better_per_vox, model_worst_per_vox
        
        # Store errors for continuous metrics
        for i, error in enumerate(errors):
            model_errors_over_time[i][time_point, :] = error
            
    #discrete metrics         
    model_better_over_time, model_worst_over_time = model_better_over_time.astype(int), model_worst_over_time.astype(int)
    best_of_all, worst_of_all = np.choose(model_better_over_time, model_vox_preds), np.choose(model_worst_over_time, model_vox_preds)
    
    #continuous metrics  
    error_difference_over_time_matrix = np.empty((len(model_vox_preds), len(model_vox_preds), num_time_points, num_voxels))
    relative_error_difference_over_time_matrix = np.empty((len(model_vox_preds), len(model_vox_preds), num_time_points, num_voxels))
    for i in range(len(model_vox_preds)):
        for j in range(len(model_vox_preds)):
            error_difference_over_time_matrix[i, j] = model_errors_over_time[j] - model_errors_over_time[i]
            if normalize:
                relative_error_difference_over_time_matrix[i, j] = error_difference_over_time_matrix[i, j] / (np.maximum(model_errors_over_time[i], model_errors_over_time[j])+1e-10)
            else:
                relative_error_difference_over_time_matrix[i, j] = error_difference_over_time_matrix[i, j]
    
    return model_better_over_time, best_of_all, worst_of_all, error_difference_over_time_matrix, relative_error_difference_over_time_matrix, model_errors_over_time

def difference_matrix_over_time_to_RGB(diff_matrix, min_val=None, max_val=None):
    r""" 
    diff_matrix : for use with things like "relative_error_difference_over_time_matrix" (shape : (len(model_vox_preds), len(model_vox_preds), num_time_points, num_voxels))
    """
    num_models, _, num_time_points, num_voxels = diff_matrix.shape
    raise NotImplementedError("This function is not yet implemented.")
    if num_models == 2:
        return diff_matrix
    
        
    if min_val is None:
        min_val = np.min(diff_matrix)
    if max_val is None:
        max_val = np.max(diff_matrix)
    
    diff_matrix = (diff_matrix - min_val) / (max_val - min_val)
    return np.stack([np.clip(1 - diff_matrix, 0, 1), np.clip(diff_matrix, 0, 1), np.zeros_like(diff_matrix)], axis=-1)  