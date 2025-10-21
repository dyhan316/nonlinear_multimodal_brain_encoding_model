import os 
import config 
#OPT
import re

import joblib 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ridge_utils
from ridge_utils.util import make_delayed

from ridge_utils.dsutils import make_word_ds
from ridge_utils.tokenization_helpers import *
import time
import os 
from tqdm import tqdm

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "./ridge_utils/"))

import argparse
import config 

from huggingface_hub import login
login("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")

# add_space_to_wordseqjbl, adhoc implementation: For use with LLAMA3, empty string needs to be added at the beginning of wordseqs... (since tokenization helper assumes there's a " " in front of each word) Some items are missing this, so add it and save updated wordseqs!
#! run this!! (Actually no need any more.. just here for recordkeeping purposes)
def update_word_seq_and_save(): 
    wordseqs = joblib.load("ridge_utils/wordseqs.jbl") # Load wordseqs containing story annotations
    for es, story in enumerate(wordseqs):
        if wordseqs[story].data[0] != '':
            wordseqs[story].data = [''] + wordseqs[story].data
            wordseqs[story].data_times = np.concatenate([[0.00623583], wordseqs[story].data_times])
            #why 0.00623583? because this was used in all other stories that start with ''
    joblib.dump(wordseqs, "ridge_utils/updated_wordseqs.jbl")

print(config.STIM_FEATURE_EXTRACTED_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, required = False, default = "huggyllama/llama-7b")
parser.add_argument('--story_partition', type = int, required=True) # From 0 to 5! In units of 20! 
args = parser.parse_args()

####for figuring out the model tokenizer id relationship (for expanding to LLAMA)
#model_list = ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
#              "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf","meta-llama/Llama-2-70b-hf",
#              "meta-llama/Meta-Llama-3-8B" , "meta-llama/Meta-Llama-3-70B"]
#for model in model_list :
#    tokenizer = AutoTokenizer.from_pretrained(model) # Same tokenizer for all sizes
#    inputs = tokenizer('I reached over 8 thereare hi my name is 8', return_tensors="pt")
#    print(f"model : {model} | tokenizer id : {inputs['input_ids']}")
#    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
#    print(tokenizer.decode(inputs['input_ids'][0]))
#    print('='*10)
#
#
#    inputs = tokenizer(' ', return_tensors="pt")
#    print(f"model : {model} | tokenizer id : {inputs['input_ids']}")
#    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
#    print(tokenizer.decode(inputs['input_ids'][0]))
#    print('='*10)
#
#    inputs = tokenizer('', return_tensors="pt")
#    print(f"model : {model} | tokenizer id : {inputs['input_ids']}")
#    print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
#    print(tokenizer.decode(inputs['input_ids'][0]))
#    print('===='*10)
####


wordseqs = joblib.load("ridge_utils/updated_wordseqs.jbl") # Load wordseqs containing story annotations

total_story_list = [story for story in list(wordseqs)]
total_len = len(total_story_list)
do_story_list = total_story_list[args.story_partition*30:(args.story_partition+1)*30] # for partitioning the stories # Change to 55!! (so it only needs to be done twice)

# do_story_list  = ['sweetaspie', 'tildeath', 'alternateithicatom'] #for debugging, 

#only take the wordseqs that we will use 
wordseqs = {story : wordseqs[story] for story in do_story_list}

# We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name) # Same tokenizer for all sizes

# Make dictionary to align tokens and words
#if model_name == "facebook/opt-125m":
#    text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_opt(wordseqs, tokenizer, 256, 512)
#elif "Llama-3" in model_name:
#    llama_version = 3
#elif "Llama-2" in model_name: 
#    llama_version = 2
#elif "huggyllama" in model_name: 
#    llama_version = 1
#else : 
#    raise NotImplementedError
text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_robust(wordseqs, tokenizer, 256, 512)
#text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_llama(wordseqs, tokenizer, 256, 512)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype = torch.float16) 
    model.eval()
    # We will extract features from the 9th layer of the model
    # LAYER_NUM = 9
    num_layers = model.config.num_hidden_layers
    text_dict3_list = [text_dict3.copy() for _ in range(num_layers)]
    start_time = time.time()
    for phrase in text_dict2:
        if text_dict2[phrase]:
            inputs = {}
            inputs['input_ids'] = torch.tensor([text_dict[phrase]]).int().to(model.device) # Load token input ids for the corresponding phrase
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(model.device)
            with torch.no_grad():
                out = list(model(**inputs, output_hidden_states=True)[2]) #list of hidden states (layer by layer)
                print("Processing phrase:", model.device)
            for layer_num in range(num_layers):
                out_layer = out[layer_num][0].cpu().detach().numpy() # Extract layer by layer! Should functionalize this and use for loop to get each layer!?
                out_layer = np.array(out_layer)
                this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
                acc_true = 0
                for ei, i in enumerate(this_key):
                    if this_key[:ei+1] in text_dict3_list[layer_num]:
                        acc_true += 1
                        text_dict3_list[layer_num][this_key[:ei+1]] = out_layer[ei,:]
    end_time = time.time()

    print("Feature extraction took", end_time - start_time, "seconds on", model.device)
    
    # Convert back from dictionary to matrix
    for layer_num in tqdm(range(num_layers)):
    # for layer_num in range(num_layers):
        feats, ds_feats = convert_to_feature_mats_llama(wordseqs, tokenizer, 256, 512, text_dict3_list[layer_num]) #, llama_version = llama_version)
        
        model_file_name = model_name.replace("/", "_")
        os.makedirs(os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, model_file_name), exist_ok = True)
        
        # Define the file paths and saving or updating!
        feats_file_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, model_file_name, f"feats_layer_{layer_num}.jbl")
        ds_feats_file_path = os.path.join(config.STIM_FEATURE_EXTRACTED_DIR, model_file_name, f"ds_feats_layer_{layer_num}.jbl")

        # Process feats
        if os.path.exists(feats_file_path):
            existing_feats = joblib.load(feats_file_path)
            existing_feats.update(feats)  # Assuming feats is a dictionary
        else:
            existing_feats = feats
        joblib.dump(existing_feats, feats_file_path)

        # Process ds_feats
        if os.path.exists(ds_feats_file_path):
            existing_ds_feats = joblib.load(ds_feats_file_path)
            existing_ds_feats.update(ds_feats)  # Assuming ds_feats is a dictionary
        else:
            existing_ds_feats = ds_feats
        joblib.dump(existing_ds_feats, ds_feats_file_path)

        # Check if all stories have been extracted
        if len(joblib.load(feats_file_path)) == total_len:
            print("ALL stories successfully extracted!!")
        else : 
            print("Some stories are missing!!")
            print(f"number of stories extracted : {len(existing_ds_feats.keys())} out of {total_len}")
        

#! backup!
""" 
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model, device_map = 'auto', torch_dtype = torch.float16) 

    model.eval()
    # We will extract features from the 9th layer of the model
    LAYER_NUM = 9

    start_time = time.time()
    for phrase in text_dict2:
        if text_dict2[phrase]:
            inputs = {}
            inputs['input_ids'] = torch.tensor([text_dict[phrase]]).int().to(model.device) #.to(device) #해당 phrase의 token input ids 불러오기 ㄴ
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(model.device)
            with torch.no_grad():
                out = list(model(**inputs, output_hidden_states=True)[2]) #list of hidden states (layer by layer)
            import pdb ; pdb.set_trace()
            out = out[LAYER_NUM][0].cpu().detach().numpy() #!여기서 layer별로 가져감! 여기를 functionalize 해서 for loop으로 layer별로 가져가도로고 하자!?
            out = np.array(out)
            this_key = tuple(inputs['input_ids'][0].cpu().detach().numpy())
            acc_true = 0
            for ei, i in enumerate(this_key):
                if this_key[:ei+1] in text_dict3:
                    acc_true += 1
                    text_dict3[this_key[:ei+1]] = out[ei,:]
    end_time = time.time()

    print("Feature extraction took", end_time - start_time, "seconds on", model.device)
    
    # Convert back from dictionary to matrix
    # feats = convert_to_feature_mats_opt(wordseqs, tokenizer, 256, 512, text_dict3)
    
    feats, ds_feats = convert_to_feature_mats_llama(wordseqs, tokenizer, 256, 512, text_dict3) #, llama_version = llama_version)
    import pdb ; pdb.set_trace()
    print(len(wordseqs['alternateithicatom'].data), wordseqs['alternateithicatom'].data_times.shape, wordseqs['alternateithicatom'].data[:10])
    print(feats['alternateithicatom'].data.shape, feats['alternateithicatom'].data_times.shape)
    print(ds_feats['alternateithicatom'].shape)
    
# 1 : {'start_token_id' : 1, 'space_token_id' : 29871, 'boundary_token_id' : 29947, 'word_start_prefix' : '▁'},


"""
