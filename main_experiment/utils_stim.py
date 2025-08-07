import os
import numpy as np
import json

import config
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed

from utils_audio import extract_speech_features 
import torchaudio
from pathlib import Path
from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel,\
                         AutoFeatureExtractor, WhisperModel
import torch
import collections
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional

from transformers import AutoTokenizer
# from LLM import encode_word
import joblib
import ridge_utils
from torch import nn

TARGET_SAMPLE_RATE = 16000 #16kHz, by default for all models!

def get_story_wordseqs(stories):
    total_wordseqs = joblib.load('./ridge_utils/updated_wordseqs.jbl')
    selected_wordseqs = {story: total_wordseqs[story] for story in stories}
    return selected_wordseqs
    
# def get_story_wordseqs(stories):
#     """loads words and word times of stimulus stories
#     """
#     #################################
#     grids = load_textgrids(stories, config.DATA_TRAIN_DIR)
#     with open(os.path.join(config.DATA_TRAIN_DIR, "respdict.json"), "r") as f:
#         respdict = json.load(f)
#     trfiles = load_simulated_trfiles(respdict)
#     wordseqs = make_word_ds(grids, trfiles)
#     return wordseqs

def sync_for_subtokens(wordseqs, features): #bjkim
    """synchronize time points with subtokenization
    """
    tokenizer = features.model.tokenizer
    #new_word : equivalent to words (but with subtokens)
    for story in wordseqs:
        print('synchronising for: ', story)
        words = wordseqs[story].data
        word_times = wordseqs[story].data_times
        new_word_times = []
        new_word = []
        for word, word_time in zip(words, word_times):
            encoded_words, _ = encode_word(word, tokenizer)
            num_subtokens = len(encoded_words)
            if num_subtokens > 1:
                subword_times = np.linspace(word_time - 0.001, word_time + 0.001, num_subtokens, endpoint = False)
                new_word_times.extend(subword_times)
                new_word.extend(encoded_words)
            elif num_subtokens == 0:
                new_word_times.append(word_time)
                new_word.append('')
            else:
                new_word_times.append(word_time)
                new_word.append(encoded_words[0])
        wordseqs[story].data_times = new_word_times
    # import pdb ; pdb.set_trace() #this will be removed later
    return wordseqs
        

def load_stim_features(stim_setting, S_SETTINGS, stories, layer, tr_stats = None, stacked = True, downsampled = True):
    """
    load stimulus features from precomputed joblib files 
    #hf_model_name : setting[HF_MODEL_NAME]
    """
    if stim_setting == "default" : 
        return None 
    #! : todo : make it so that if not downsampling, than make_delayed or sth is not done (and trimming also not done!)
    
    #loading precomputed stuff
    if S_SETTINGS['audio_or_LLM'] == 'LLM' :
        hf_model_name = S_SETTINGS['HF_MODEL_NAME']
        model_file_name = hf_model_name.replace("/", "_")
        
        #word_vecs : per token features, but not used when doing downsampling thing 
        word_vecs_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, model_file_name, f"feats_layer_{layer}.jbl")
        word_vecs_all= joblib.load(word_vecs_path)
        word_vecs = {story: word_vecs_all[story] for story in stories if story in word_vecs_all}
                
        ds_vecs_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, model_file_name, f"ds_feats_layer_{layer}.jbl")
        ds_vecs_all = joblib.load(ds_vecs_path)
        ds_vecs = {story: ds_vecs_all[story] for story in stories if story in ds_vecs_all}
        
    elif S_SETTINGS['audio_or_LLM'] == 'audio' :
        word_seqs = get_story_wordseqs(stories)
        
        # #! my thing!!!! ==> 일단은 다른곳에 해자!
        # get_stim_audio_for_training(stories, S_SETTINGS)
        # exit()
        
        audio_feat_vecs = get_stim_audio(stories, S_SETTINGS) 
        ds_vecs = {story : lanczosinterp2D(audio_feat_vecs[story].data, audio_feat_vecs[story].data_times, 
                                           word_seqs[story].tr_times) for story in stories}        

    #below : post-processing stuff that was originally done in get_stim 
    if stim_setting.startswith("DSNFT") :
        ds_vecs = {story : ds_vecs[story][:-config.S_SETTINGS['TRIM_STIM_LAST_WORDS']] for story in stories} #only remove the last five cuz endpoints don't match b/w resp and stim
        ds_mat = np.vstack([ds_vecs[story] for story in stories])  
    else : 
        ds_mat = np.vstack([ds_vecs[story][5+config.TRIM:-config.TRIM] for story in stories]) #DO NOT remove! required for timepoint matching!
    
    if tr_stats is None: 
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else: 
        r_mean, r_std = tr_stats
    
    #if DSNFT, terminate early without doing make_delayed (i.e. don't stack the past 4 things)
    if stim_setting.startswith("DSNFT") :
        ds_vecs = {story : np.nan_to_num(np.dot((ds_vecs[story] - r_mean), np.linalg.inv(np.diag(r_std)))) for story in stories}
        if tr_stats is None: return ds_vecs, (r_mean, r_std), (0, 0) #word_mean, word_std not used, so just return 0
        else : return ds_vecs
            
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS) #stack past 4 things!

    if stacked: #original behavior
        if tr_stats is None: return del_mat, (r_mean, r_std), (0, 0) #word_mean, word_std not used, so just retunr 0
        else: return del_mat
    else : #ADDED by danny #return dict #참조 : 원래는 이것 결과가 stacked = True일때의 결과와 다름 (cuz np.nan_to_num안에 있는 np.dot 이런것들 때문에 !!  일단은 그냥 stacked경우에서 다시 story별로 잘라내는 걸로 퉁침!!
        #individually cutup the del_mat, and so on for each story, then return. use np.split
        # import pdb ; pdb.set_trace()
        # Calculate the lengths of the chunks
        lengths = [len(ds_vecs[story][5+config.TRIM:-config.TRIM]) for story in stories] #TRIM CUZ NEEDED to account for the TRIM in the original code

        # Calculate the indices where the array should be split
        indices = np.cumsum(lengths)#[:-1]

        # Split the array and make into dict (ex : {story1 : del_mat[0 : indices[0]], story2 : del_mat[indices[0] : indices[1]], ...})
        # Assume stories is a list of story names and indices is a list of indices
        del_mat_dict = {}
        for i, story in enumerate(stories):
            if i == 0:
                del_mat_dict[story] = del_mat[0:indices[0]]
            else:
                del_mat_dict[story] = del_mat[indices[i-1]:indices[i]]
        if tr_stats is None: return del_mat_dict, (r_mean, r_std), (0, 0) #word_mean, word_std not used, so just retunr 0
        else: return del_mat_dict    
    
def get_stim(stim_setting, S_SETTINGS, stories, features, tr_stats = None, stacked = True):
    # outdated, probably not used anymore since precomputation will be done!
    """extract quantitative features of stimulus stories
    """
    #TODO : implement no resampling, and no trimming and so on!! (TRIMMING 에서 5+TRIM : -TRIM의 뜻을 유념하며, 그것을 바꾸기)
    #TODO : becareful : TRIM값이 5이고 5+TRIM : -TRIM으로 해야, resp와 shape가 같다!! (아마 response가 미리 짤린듯)) #!#! 주의 주의
    if stim_setting == "ds_no_front_trim" : 
        raise NotImplementedError("THIS IS NO LONGER SUPPORTED... use load_stim_features thign... GPT로는 굳이 되도록 할 이유가 없어서, audio model and LLAMA에서만 되도록 함!")
        # return get_ds_no_front_trim(stim_setting, S_SETTINGS, stories, features, tr_stats, stacked) #adhoc, needs to be changed later
    word_seqs = get_story_wordseqs(stories)
    word_seqs = sync_for_subtokens(word_seqs, features) if S_SETTINGS['S_MODEL'] != "GPT1" else word_seqs #CHANGE THIS
    word_vecs = {story : features.make_stim(word_seqs[story].data) for story in stories}
    word_mat = np.vstack([word_vecs[story] for story in stories])
    word_mean, word_std = word_mat.mean(0), word_mat.std(0)
    
    #ds : downsampled(=lanczos interpolated)
    ds_vecs = {story : lanczosinterp2D(word_vecs[story], word_seqs[story].data_times, word_seqs[story].tr_times) 
               for story in stories}
    ds_mat = np.vstack([ds_vecs[story][5+config.TRIM:-config.TRIM] for story in stories])
    if tr_stats is None: 
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else: 
        r_mean, r_std = tr_stats
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    if stacked: #original behavior
        if tr_stats is None: return del_mat, (r_mean, r_std), (word_mean, word_std)
        else: return del_mat
    else : #ADDED by danny #return dict
        #참조 : 원래는 이것 결과가 stacked = True일때의 결과와 다름 (cuz np.nan_to_num안에 있는 np.dot 이런것들 때문에 !!  일단은 그냥 stacked경우에서 다시 story별로 잘라내는 걸로 퉁침!!
        #individually cutup the del_mat, and so on for each story, then return. use np.split
        
        # Calculate the lengths of the chunks
        lengths = [len(ds_vecs[story][5+config.TRIM:-config.TRIM]) for story in stories] #TRIM CUZ NEEDED to account for the TRIM in the original code

        # Calculate the indices where the array should be split
        indices = np.cumsum(lengths)#[:-1]

        # Split the array and make into dict (ex : {story1 : del_mat[0 : indices[0]], story2 : del_mat[indices[0] : indices[1]], ...})
        # Assume stories is a list of story names and indices is a list of indices
        del_mat_dict = {}
        for i, story in enumerate(stories):
            if i == 0:
                del_mat_dict[story] = del_mat[0:indices[0]]
            else:
                del_mat_dict[story] = del_mat[indices[i-1]:indices[i]]
        if tr_stats is None: return del_mat_dict, (r_mean, r_std), (word_mean, word_std)
        else: return del_mat_dict
        

def get_stim_non_interp(stories, features, tr_stats = None):
    """extract quantitative features of stimulus stories without Lanczos interpolation
    """
    raise NotImplementedError("not implemented yet, DO IT !!")

def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time
    """
    delresp = make_delayed(resp[:, vox], config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)

def predict_word_times(word_rate, resp, starttime = 0, tr = 2):
    """predict evenly spaced word times from word rate
    """
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):  
        if num < 1: continue
        word_times.extend(np.linspace(mid - half, mid + half, num, endpoint = False) + half / num)
    return np.array(word_times), tr_times


def get_stim_audio(stories, S_SETTINGS):
    """extract quantitative features of stimulus audio
    """
    stimulus_dir = Path(config.DATA_AUDIO_DIR)
    
    # chunk size (stride), and contextsz (audio model window) in miliseconds! 
    #(i.e. chunksz 100 : 100ms =  0.1s and so on )
    chunksz = S_SETTINGS['chunksz']
    contextsz = S_SETTINGS['contextsz']
    chunksz_sec = chunksz / 1000. #convert from ms to s 

    # context size in terms of chunks
    assert (contextsz % chunksz) == 0, "These must be divisible"
    contextsz_sec = contextsz / 1000.

    model_name = S_SETTINGS['S_MODEL']
    with open('speech_model_configs.json', 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']
    print('Loading model', model_name, 'from the Hugging Face Hub...')
    if torch.cuda.is_available():
        print("if GPU is available, then will assume feature extraction should be done and hence will load model and feature_extractor ")
        model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=True).cuda()
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
    else : 
        print("if GPU is not available, then will assume feature extraction ALREADY DONE")
        model = nn.Identity() #dummy model
        feature_extractor = None
    
    stimulus_paths: Dict[str, Path] = {} # map of stimulus name --> file path. We also use this as the list of stimuli

    for story in stories:
        for ext in ['flac', 'wav']:
            stimulus_path = stimulus_dir / f"{story}.{ext}"
            if stimulus_path.exists() and stimulus_path.is_file():
                stimulus_paths[story] = stimulus_path
                break

    missing_stories = set(stories).difference(set(stimulus_paths.keys()))
    if len(missing_stories) > 0:
        raise RuntimeError(f"missing stimuli for stories: " + ' '.join(missing_stories))

    assert len(stimulus_paths) > 0, "no stimuli to process!"

    # Make sure that all preprocessed stimuli exist and are readable.
    for stimulus_name, stimulus_local_path in stimulus_paths.items():
        wav, sample_rate = torchaudio.load(stimulus_local_path)

    # audio_save_dir = Path(config.S_SETTINGS['AUDIO_FEAT_DIR'])
    # model_save_path = f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}/{model_name}"
    model_save_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, 
                                   f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}/{model_name}")
    
    print('Saving features to:', model_save_path)

    ## Feature extraction loop
    # Go through each stimulus and save resulting features
    torch.set_grad_enabled(False) # VERY important! (for memory)
    model.eval()

    concat_feats = dict()
    concat_times = dict()

    stimulus_paths = collections.OrderedDict(stimulus_paths.items())
    for stimulus_name, stimulus_local_path in tqdm(stimulus_paths.items(), desc='Processing stories'):
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        
        if wav.shape[0] != 1: wav = wav.mean(0, keepdims=True) # convert to mono
        if sample_rate != TARGET_SAMPLE_RATE: # resample to 16 kHz
            #즉, audio sampling rate : 16k는 고정!! (즉, chunksz는 알아서 16k에 맞춰서 resampling이 된다! )
            wav = torchaudio.functional.resample(wav, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        wav.squeeze_(0) # shape: (num_samples,)

        assert sample_rate == TARGET_SAMPLE_RATE, f"Expected sample rate {TARGET_SAMPLE_RATE} but got {sample_rate}"

        features_save_path = os.path.join(model_save_path, stimulus_name)
        times_save_path = f"{features_save_path}_times"

        ### do not overwrite
        if os.path.exists(times_save_path + '.npz'):
            print(f"Skipping {stimulus_name}, timestamps found at {times_save_path}")

        else:
            # Call a separate function to do the actual feature extraction
            extract_features_kwargs = {
                'model': model, 'model_config': model_config,
                'wav': wav.to(model.device),
                'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
                'sel_layers': None, 'feature_extractor': feature_extractor,  #sel_layers : dud (by default extracts and saves ALL layers)
                'require_full_context': False,
                'batchsz': 64, 'return_numpy': False #! 원래 64, increase as needed  # 64 : for whisper-large, 450 for whipser-tiny
            } 

            extracted_features = extract_speech_features(**extract_features_kwargs)
            out_features, times, hidden_features = [extracted_features[k] for k in \
                                                    ['final_outputs', 'times', 'module_features']]
            del extracted_features # free up some memory after we've selected the outputs we want; maybe unnecessary

            # saving final output layer and times
            os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
            np.savez_compressed(features_save_path + '.npz', features=out_features.numpy())
            np.savez_compressed(times_save_path + '.npz', times=times.numpy())
            
            # saving hiddne feautres
            for layer_name, layer_features in hidden_features.items():
                layer_save_path = os.path.join(model_save_path, layer_name) 
                os.makedirs(layer_save_path, exist_ok=True)
                hidden_feature_save_path = os.path.join(layer_save_path, stimulus_name)
                np.savez_compressed(hidden_feature_save_path+ '.npz', features=layer_features.numpy())
                
        #====modified code start ====#
        #! config에서 final이면 int(XXX[1])하지말고, "final"로 return하기? (then if final thing ... )
        if S_SETTINGS['FEATURE_LAYER'] == "final":
            features = np.load(features_save_path + '.npz')['features']
        elif isinstance(S_SETTINGS['FEATURE_LAYER'], int) : 
            if 'whisper' in model_name: # then encoder.L 이다
                layer_path = os.path.join(model_save_path,f"encoder.{S_SETTINGS['FEATURE_LAYER']}")
            else : #then layer.L 이다
                layer_path = os.path.join(model_save_path,f"layer.{S_SETTINGS['FEATURE_LAYER']}")    
            features = np.load(os.path.join(layer_path, stimulus_name) + '.npz')['features']
        elif isinstance(S_SETTINGS['FEATURE_LAYER'], list) :
            features_list = []
            for layer in S_SETTINGS['FEATURE_LAYER']:
                if 'whisper' in model_name: # then encoder.L 이다
                    layer_path = os.path.join(model_save_path,f"encoder.{layer}")
                else : #then layer.L 이다
                    layer_path = os.path.join(model_save_path,f"layer.{layer}")
                features_list.append(np.load(os.path.join(layer_path, stimulus_name) + '.npz')['features'])
            features = np.concatenate(features_list, axis = 1)
        else : 
            raise ValueError("S_SETTINGS['FEATURE_LAYER'] should be either 'final' or int")

        # out_features = np.load(features_save_path + '.npz')['features']
        times = np.load(times_save_path + '.npz')['times']

        concat_feats[stimulus_name] = [features, times]
    torch.set_grad_enabled(True) #!  VERY important! (sets back to default torch behavior)
    return create_audio_feat_dict(concat_feats)

class audio_DataSequence:
    def __init__(self, data=None, data_times=None, time_windows=None):
        self.data = data
        self.data_times = data_times
        self.time_windows = time_windows


def create_audio_feat_dict(data_source):
    audio_thing_dict = {}
    for stimulus_name, value in data_source.items():
        # Create an instance of audio_DataSequence for each stimulus
        audio_data_sequence = audio_DataSequence()
        audio_data_sequence.data, audio_data_sequence.time_windows = value
        if audio_data_sequence.time_windows.ndim > 1:
            audio_data_sequence.data_times = audio_data_sequence.time_windows[:, 1]
        else:
            audio_data_sequence.data_times = audio_data_sequence.time_windows
        # Assign the instance to the dictionary with stimulus_name as the key
        audio_thing_dict[stimulus_name] = audio_data_sequence
    return audio_thing_dict


from utils_audio import forward_speech_model
def get_stim_audio_for_training(stories, S_SETTINGS):
    """extract quantitative features of stimulus audio
    """
    stimulus_dir = Path(config.DATA_AUDIO_DIR)
    
    # chunk size (stride), and contextsz (audio model window) in miliseconds! 
    #(i.e. chunksz 100 : 100ms =  0.1s and so on )
    chunksz = S_SETTINGS['chunksz']
    contextsz = S_SETTINGS['contextsz']
    chunksz_sec = chunksz / 1000. #convert from ms to s 

    # context size in terms of chunks
    assert (contextsz % chunksz) == 0, "These must be divisible"
    contextsz_sec = contextsz / 1000.

    model_name = S_SETTINGS['S_MODEL']
    with open('speech_model_configs.json', 'r') as f:
        model_config = json.load(f)[model_name]
        model_hf_path = model_config['huggingface_hub']
    print('Loading model', model_name, 'from the Hugging Face Hub...')
    import pdb ; pdb.set_trace()
    if torch.cuda.is_available():
        print("if GPU is available, then will assume feature extraction should be done and hence will load model and feature_extractor ")
        model = AutoModel.from_pretrained(model_hf_path, output_hidden_states=False).cuda() #! changed to output_hidden_states = False!
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_hf_path)
    else : 
        raise ValueError("GPU should be available!")
        # print("if GPU is not available, then will assume feature extraction ALREADY DONE")
        # model = nn.Identity() #dummy model
        # feature_extractor = None
        
    stimulus_paths: Dict[str, Path] = {} # map of stimulus name --> file path. We also use this as the list of stimuli

    for story in stories:
        for ext in ['flac', 'wav']:
            stimulus_path = stimulus_dir / f"{story}.{ext}"
            if stimulus_path.exists() and stimulus_path.is_file():
                stimulus_paths[story] = stimulus_path
                break

    missing_stories = set(stories).difference(set(stimulus_paths.keys()))
    if len(missing_stories) > 0:
        raise RuntimeError(f"missing stimuli for stories: " + ' '.join(missing_stories))

    assert len(stimulus_paths) > 0, "no stimuli to process!"

    # Make sure that all preprocessed stimuli exist and are readable.
    for stimulus_name, stimulus_local_path in stimulus_paths.items():
        wav, sample_rate = torchaudio.load(stimulus_local_path)

    # audio_save_dir = Path(config.S_SETTINGS['AUDIO_FEAT_DIR'])
    #! remove model_save_path since remove it?
    model_save_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, 
                                   f"features_cnk{chunksz_sec:0.1f}_ctx{contextsz_sec:0.1f}/{model_name}")
    
    print('Saving features to:', model_save_path)

    ## Feature extraction loop
    # Go through each stimulus and save resulting features
    #!========removed this part=================
    # torch.set_grad_enabled(False) # VERY important! (for memory)
    # model.eval()
    #!==========================================
    torch.set_grad_enabled(True)
    model.train() #! maybe change this?    

    concat_feats = dict()
    concat_times = dict()

    stimulus_paths = collections.OrderedDict(stimulus_paths.items())
    for stimulus_name, stimulus_local_path in tqdm(stimulus_paths.items(), desc='Processing stories'):
        wav, sample_rate = torchaudio.load(stimulus_local_path)
        
        if wav.shape[0] != 1: wav = wav.mean(0, keepdims=True) # convert to mono
        if sample_rate != TARGET_SAMPLE_RATE: # resample to 16 kHz
            #즉, audio sampling rate : 16k는 고정!! (즉, chunksz는 알아서 16k에 맞춰서 resampling이 된다! )
            wav = torchaudio.functional.resample(wav, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        wav.squeeze_(0) # shape: (num_samples,)

        assert sample_rate == TARGET_SAMPLE_RATE, f"Expected sample rate {TARGET_SAMPLE_RATE} but got {sample_rate}"

        features_save_path = os.path.join(model_save_path, stimulus_name)
        times_save_path = f"{features_save_path}_times"

        # Call a separate function to do the actual feature extraction
        extract_features_kwargs = {
            'model': model, 'model_config': model_config,
            'wav': wav.to(model.device),
            'chunksz_sec': chunksz_sec, 'contextsz_sec': contextsz_sec,
            'sel_layers': None, 'feature_extractor': feature_extractor,  #sel_layers : dud (by default extracts and saves ALL layers)
            'require_full_context': False,
            'batchsz': 8, 'return_numpy': False, #! 원래 64, increase as needed  # 64 : for whisper-large, 450 for whipser-tiny
            'move_to_cpu' : False, #! added by danny
        } 
        import pdb ; pdb.set_trace()
        extracted_features = forward_speech_model(**extract_features_kwargs)
        # extracted_features = extract_speech_features(**extract_features_kwargs)
        import pdb ; pbb.set_trace()
        # extracted_features = extract_speech_features(**extract_features_kwargs)
        out_features, times, hidden_features = [extracted_features[k] for k in \
                                                ['final_outputs', 'times', 'module_features']]
        del extracted_features # free up some memory after we've selected the outputs we want; maybe unnecessary

        # saving final output layer and times
        os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
        np.savez_compressed(features_save_path + '.npz', features=out_features.numpy())
        np.savez_compressed(times_save_path + '.npz', times=times.numpy())
        
        # saving hiddne feautres
        for layer_name, layer_features in hidden_features.items():
            layer_save_path = os.path.join(model_save_path, layer_name) 
            os.makedirs(layer_save_path, exist_ok=True)
            hidden_feature_save_path = os.path.join(layer_save_path, stimulus_name)
            np.savez_compressed(hidden_feature_save_path+ '.npz', features=layer_features.numpy())
                
        #====modified code start ====#
        #! config에서 final이면 int(XXX[1])하지말고, "final"로 return하기? (then if final thing ... )
        if S_SETTINGS['FEATURE_LAYER'] == "final":
            features = np.load(features_save_path + '.npz')['features']
        elif isinstance(S_SETTINGS['FEATURE_LAYER'], int) : 
            if 'whisper' in model_name: # then encoder.L 이다
                layer_path = os.path.join(model_save_path,f"encoder.{S_SETTINGS['FEATURE_LAYER']}")
            else : #then layer.L 이다
                layer_path = os.path.join(model_save_path,f"layer.{S_SETTINGS['FEATURE_LAYER']}")    
            features = np.load(os.path.join(layer_path, stimulus_name) + '.npz')['features']
        elif isinstance(S_SETTINGS['FEATURE_LAYER'], list) :
            features_list = []
            for layer in S_SETTINGS['FEATURE_LAYER']:
                if 'whisper' in model_name: # then encoder.L 이다
                    layer_path = os.path.join(model_save_path,f"encoder.{layer}")
                else : #then layer.L 이다
                    layer_path = os.path.join(model_save_path,f"layer.{layer}")
                features_list.append(np.load(os.path.join(layer_path, stimulus_name) + '.npz')['features'])
            features = np.concatenate(features_list, axis = 1)
        else : 
            raise ValueError("S_SETTINGS['FEATURE_LAYER'] should be either 'final' or int")

        # out_features = np.load(features_save_path + '.npz')['features']
        times = np.load(times_save_path + '.npz')['times']

        concat_feats[stimulus_name] = [features, times]
    torch.set_grad_enabled(True) #!  VERY important! (sets back to default torch behavior)
    return create_audio_feat_dict(concat_feats)

        
"""  
밑의 것에서 보듯이, tr_times가 잘 맞다!! (audio and LLM are same.. no need to worry)

(Pdb) word_vecs['souls'].data_times[-10:]
array([715.92040816, 716.17482993, 717.75317127, 718.42970522,
    718.67414966, 718.92358277, 719.38976465, 720.12949451,
    721.55759637, 721.83197279])
(Pdb) word_vecs['souls'].tr_times[-10:]
array([711., 713., 715., 717., 719., 721., 723., 725., 727., 729.])

(Pdb) word_seqs['souls'].tr_times[-10:]
(Pdb) audio_feat_vecs['souls'].data_times[-10:]
array([729.1, 729.2, 729.3, 729.4, 729.5, 729.6, 729.7, 729.8, 729.9,
    730. ], dtype=float32)
array([711., 713., 715., 717., 719., 721., 723., 725., 727., 729.])
"""
