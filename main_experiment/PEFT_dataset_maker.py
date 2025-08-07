import os 
import config 
#OPT

import joblib 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import ridge_utils
from ridge_utils.util import make_delayed

from ridge_utils.dsutils import make_word_ds
from ridge_utils.tokenization_helpers import generate_efficient_feat_dicts_llama, convert_to_feature_mats_llama
import time
import os 
from tqdm import tqdm

import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), "./ridge_utils/"))

import argparse
import config
# from ridge_utils.DataSequence import DataSequence
from utils_PEFT import PEFT_DataSequence
from utils_resp import get_resp

import numpy as np
import itertools as itools
from ridge_utils.interpdata import sincinterp2D, gabor_xfm2D, lanczosinterp2D


#########SETUP###############
###for figuring out the model tokenizer id relationship (for expanding to LLAMA)
model_list = ["huggyllama/llama-7b", "huggyllama/llama-13b", "huggyllama/llama-30b", "huggyllama/llama-65b",
              "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf","meta-llama/Llama-2-70b-hf", 
              "meta-llama/Meta-Llama-3-8B" , "meta-llama/Meta-Llama-3-70B"]

wordseqs = joblib.load("ridge_utils/updated_wordseqs.jbl") # Load wordseqs containing story annotations
#############################
 
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, required = False, default = "huggyllama/llama-7b")
parser.add_argument("--subject", type = str, required = True)
parser.add_argument("--resp_setting", type = str, default = "default", 
                        choices = ["default",
                                   "default_32","default_128","default_256","default_512","default_1024", "default_2048", "default_4096", "default_8192", "default_16384", "default_32768",
                                   "PCA_10", "PCA_32","PCA_128", "PCA_256","PCA_1024", "PCA_512","PCA_2048", "PCA_4096","PCA_8192","PCA_16384","PCA_32768",
                                   "PCAonIMP_10", "PCAonIMP_32","PCAonIMP_128", "PCAonIMP_512", "PCAonIMP_1024", "PCAonIMP_2048", "PCAonIMP_4096","PCAonIMP_8192"])
args = parser.parse_args()


do_story_list = eval(f"config.{args.subject}_train_list")  #to only do the stories we are supposed to do!

# #! below : overwrite for testing purposes
# do_story_list = ['sweetaspie', 'itsabox']

#only take the wordseqs that we will use 
wordseqs = {story : wordseqs[story] for story in do_story_list}

# We will be using a sliding context window with minimum size 256 words that increases until size 512 words.
model_name = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_name) # Same tokenizer for all sizes

if "Llama-3" in model_name:
    llama_version = 3
elif "Llama-2" in model_name: 
    llama_version = 2
elif "huggyllama" in model_name: 
    llama_version = 1
else : 
    raise NotImplementedError
# import pdb ; pdb.set_trace() 

text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_llama(wordseqs, tokenizer, 256, 512, llama_version = llama_version)

#! ==========below : LLM의 window를 논문 처럼 256~512말고 크게하고싶을떄 uncomment해서 쓰기!!!!===========
#! maximum size of the words : 4224... (breakingupintheageofgoogle) (story partition number 1)
# max_context_size = 2048 if model_name.startswith("huggy") else 4096 if model_name.startswith('meta-llama/Llama-2') else 8192 
# text_dict, text_dict2, text_dict3 = generate_efficient_feat_dicts_llama(wordseqs, tokenizer, 
#                                                                         int(max_context_size/2)-5, int(max_context_size/2), 
#                                                                         llama_version = llama_version)


resp_dict = get_resp(args.resp_setting, args.subject, do_story_list, stack = False) #! stack = False : we want the responses as is!

peft_dataseq_dict = {}
for story in do_story_list:
    dataseq = wordseqs[story]
    keys_to_see = []
    for key in text_dict.keys():
        if key[0] == story:
            keys_to_see.append(key)
    
    tokens_list = []
    for key in keys_to_see : 
        tokens_list.append(text_dict[key])
        
    peft_dataseq_dict[story] = PEFT_DataSequence(data = tokens_list, data_times = dataseq.data_times, 
                                                 tr_times = dataseq.tr_times, resp_data = resp_dict[story])
    
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype = torch.float16) 
model.eval() 
#! single batch일때 가정하고 돌린 것 
for story, dataseq in peft_dataseq_dict.items():
    for i, model_input_wordwise in enumerate(dataseq.data):
        inputs = {}
        inputs['input_ids'] = torch.tensor([model_input_wordwise]).int().to(model.device) 
        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape).to(model.device)
        with torch.no_grad(): #! training한다면 바뀌어야할수도? (along with loss computation and stuff)
            import pdb ; pdb.set_trace()
            out = model(**inputs, output_hidden_states=True)[0] #list of hidden states (layer by layer) #! this is where teh dmoel forward is done I think!! 
            #! notice that not the whole context is used!! (512 "words" at max))
            print(out.shape, '\n',"-="*10)
