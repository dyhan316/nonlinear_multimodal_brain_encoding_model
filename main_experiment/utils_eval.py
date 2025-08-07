import os
import numpy as np
import json

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel

import matplotlib.pyplot as plt


# from jiwer import wer
# from datasets import load_metric
# from bert_score import BERTScorer


######################BELOW : added ,new things######################
def np_mse_loss(y_true, y_pred, axis=None):
    """
    Compute the Mean Squared Error (MSE) between the true and predicted values along a specified axis.
    
    Parameters:
    - y_true: numpy.ndarray, the true target values.
    - y_pred: numpy.ndarray, the predicted values.
    - axis: int, optional, the axis along which to compute the mean squared error. Default is None, which computes the MSE over the entire array.
    
    Returns:
    - mse: numpy.ndarray or float, the computed MSE loss. If axis is None, a single float is returned. Otherwise, an array of MSE values along the specified axis is returned.
    """
    squared_diff = (y_pred - y_true) ** 2
    mse = np.mean(squared_diff, axis=axis)
    return mse

def np_mae_loss(y_true, y_pred, axis=None):
    """
    Compute the Mean Absolute Error (MAE) between the true and predicted values along a specified axis.
    
    Parameters:
    - y_true: numpy.ndarray, the true target values.
    - y_pred: numpy.ndarray, the predicted values.
    - axis: int, optional, the axis along which to compute the mean absolute error. Default is None, which computes the MAE over the entire array.
    
    Returns:
    - mae: numpy.ndarray or float, the computed MAE loss. If axis is None, a single float is returned. Otherwise, an array of MAE values along the specified axis is returned.
    """
    abs_diff = np.abs(y_pred - y_true)
    mae = np.mean(abs_diff, axis=axis)
    return mae

def spe_and_cc_norm(orig_data, data_pred, data_norm=True, max_flooring=None):
    '''
    Computes the signal power explained and the cc_norm of a model given the observed and predicted values
    Assumes normalization unless data_norm is set to False
    
    orig_data: 3D numpy array (trials, timepoints, voxels)
    
    data_pred: 2D numpy array (timepoints, voxels)
    
    data_norm: bool -> Set to False if not pre-normalized
    
    max_flooring: None/float (0-1) -> If not None, compute cc_norm in an alternate way that floors cc_max by max_flooring.
    This is helpful to clean up bad voxels that are not at all language selective.
    
    According to Schoppe: https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full
    '''
    y = np.mean(orig_data, axis=0)
    num_trials = len(orig_data)
    if not data_norm:
        variance_across_time = np.var(orig_data, axis=1, ddof=1)
        TP = np.mean(variance_across_time, axis=0)
    else:
        TP = np.zeros(orig_data.shape[2]) + 1
    SP = (1 / (num_trials-1)) * ((num_trials * np.var(y, axis=0, ddof=1)) - TP) 
    SPE_num = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1)) 
    SPE = (np.var(y, axis=0, ddof=1) - np.var(y - data_pred, axis=0, ddof=1)) / SP
    y_flip = np.swapaxes(y, axis1=0, axis2=1)
    data_flip = np.swapaxes(data_pred, axis1=0, axis2=1)
    covs = np.zeros(y_flip.shape[0])
    for i, row in enumerate(y_flip):
        covs[i] = np.cov(y_flip[i], data_flip[i])[0][1]
        
    cc_norm =  np.sqrt(1/SP) * (covs / np.sqrt(np.var(data_pred, axis=0, ddof=1))) # Warning may occur here when SP<0, but SP is signal power, and nan values are replaced with 0 during cc_max calculation
    cc_max = None
    if max_flooring is not None:
        cc_max = np.nan_to_num(1 / (np.sqrt(1 + ((1/num_trials) * ((TP/SP)-1)))))
        cc_max = np.maximum(cc_max, np.zeros(cc_max.shape) + max_flooring) #! TODO : see if this should be uncommented!! (probably should be, or else no point in doing cc_max)
        corrs = np.zeros(y_flip.shape[0])
        for i, row in enumerate(y_flip):
            corrs[i] = np.corrcoef(y_flip[i], data_flip[i])[0][1]
        cc_norm = corrs / cc_max
    elif max_flooring is None : 
        raise NotImplementedError("max_flooring is not None")
    
    return SPE, cc_norm, cc_max, corrs

def compute_r_single_trial(single_trial_data, data_pred):    
    """ 
    input 
    * single_trial_data : 2D numpy array (timepoints, voxels)
    * data_pred : 2D numpy array (timepoints, voxels)
    
    output
    * vox_wise_corr : 1D numpy array (voxels) (correlation coefficient for each voxel)
    # Don't flatten voxels!
    """
    vox_wise_corr = np.zeros(single_trial_data.shape[1])
    for vox in range(single_trial_data.shape[1]):
        vox_wise_corr[vox] = np.corrcoef(single_trial_data[:,vox], data_pred[:,vox])[0][1]
    return vox_wise_corr    

def extract_certain_area(subject, roi, arr) : 
    """
    input 
    * subject : the subvject
    * roi : roi to see (ex : speech, auditory, ... )
    * arr : ND numpy array (*, voxels)
    
    output 
    * arr : ND numpy array (*, roi voxels)
    """
    raise NotImplementedError("not implemented yet")

def plot_corrs_all_trials(all_trial_test_resp, model_vox_pred, vox, save_location = None ) :
    fig, ax1 = plt.subplots(figsize=(6,4))
    data = [i.T[vox] for i in all_trial_test_resp]

    for i in all_trial_test_resp:
        plt.plot(i.T[i], alpha=0.12, color='black')
    mean_data = np.mean(data, axis=0)
    plt.plot(mean_data, 'k--', alpha=0.8)
    plt.plot(model_vox_pred.T[vox], color='red', alpha=0.8)
    plt.xlabel("TR", size=16)
    plt.ylabel("Z-scored Response", size=16)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    
    if save_location is not None : 
        plt.savefig(save_location)    

#for converting config into a format that can be saved by npz (i.e. dict) (for loading settings easily during analysis)
def config_into_dict_for_saving(config):
    # Convert config to dictionary if it's a module
    if not isinstance(config, dict):
        config_dict = vars(config)  
    else : 
        config_dict = config
        
    # Filter out non-string, non-integer, and non-float, and os on items
    config_dict = {k: v for k, v in config_dict.items() if isinstance(v, (str, int, float, bool))} #, list, dict, tuple, set ))  and not isinstance(v, type) }

    return config_dict
    # Convert dictionary to structured numpy array
    # structured_array = np.array(list(config.items()), dtype=[('key', 'U100'), ('value', 'U100')])

    # return structured_array

##TODO : remember, tensessions's first trial : the single test trial thing!hzxvbnn,,bvcxtroiuy;


######################BELOW : original code, for running decoder######################
BAD_WORDS_PERCEIVED_SPEECH = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])
BAD_WORDS_OTHER_TASKS = frozenset(["", "sp", "uh"])

from utils_ridge.textgrid import TextGrid
def load_transcript(experiment, task):
    if experiment in ["perceived_speech", "perceived_multispeaker"]: skip_words = BAD_WORDS_PERCEIVED_SPEECH
    else: skip_words = BAD_WORDS_OTHER_TASKS
    grid_path = os.path.join(config.DATA_TEST_DIR, "test_stimulus", experiment, task.split("_")[0] + ".TextGrid")
    transcript_data = {}
    with open(grid_path) as f: 
        grid = TextGrid(f.read())
        if experiment == "perceived_speech": transcript = grid.tiers[1].make_simple_transcript()
        else: transcript = grid.tiers[0].make_simple_transcript()
        transcript = [(float(s), float(e), w.lower()) for s, e, w in transcript if w.lower().strip("{}").strip() not in skip_words]
    transcript_data["words"] = np.array([x[2] for x in transcript])
    transcript_data["times"] = np.array([(x[0] + x[1]) / 2 for x in transcript])
    return transcript_data

"""windows of [duration] seconds at each time point"""
def windows(start_time, end_time, duration, step = 1):
    start_time, end_time = int(start_time), int(end_time)
    half = int(duration / 2)
    return [(center - half, center + half) for center in range(start_time + half, end_time - half + 1) if center % step == 0]

"""divide [data] into list of segments defined by [cutoffs]"""
def segment_data(data, times, cutoffs):
    return [[x for c, x in zip(times, data) if c >= start and c < end] for start, end in cutoffs]

"""generate null sequences with same times as predicted sequence"""
def generate_null(pred_times, gpt_checkpoint, n):
    
    # load language model
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)
    
    # generate null sequences
    null_words = []
    for _count in range(n):
        decoder = Decoder(pred_times, 2 * config.EXTENSIONS)
        for sample_index in range(len(pred_times)):
            ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
            beam_nucs = lm.beam_propose(decoder.beam, ncontext)
            for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1: continue
                extend_words = [hyp.words + [x] for x in nuc]
                likelihoods = np.random.random(len(nuc))
                local_extensions = [Hypothesis(parent = hyp, extension = x) 
                                    for x in zip(nuc, logprobs, [np.zeros(1) for _ in nuc])]
                decoder.add_extensions(local_extensions, likelihoods, nextensions)
            decoder.extend(verbose = False)
        null_words.append(decoder.beam[0].words)
    return null_words
