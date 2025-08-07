import numpy as np
import logging
import sys
import joblib
import matplotlib.pyplot as plt
import torch
from ridge_utils.DataSequence import DataSequence
from transformers import AutoTokenizer, AutoModelForCausalLM


#! remove later 
import transformers
import os 
os.environ['HF_HOME'] = "/path/to/huggingface/cache"


### Warning, you are entering tokenization hell.

def compute_correct_tokens_llama(acc, acc_lookback, acc_offset, total_len):
    new_tokens = [1]
    acc_count_all = 0
    first_word = max(0,acc_offset-acc_lookback)
    last_word = min(acc_offset+1, total_len)
    acc_start = 0
    while acc_start != first_word + 1:
        if acc[acc_count_all] == 29947:
            acc_start += 1
            acc_count_all += 1
        else:
            acc_count_all += 1
    acc2 = acc[acc_count_all:]
    acc_count8 = 0
    acc_count_all = 0
    while acc_count8 != (last_word - first_word):
        if acc2[acc_count_all] == 29947:
            acc_count8 += 1
            acc_count_all += 1
        else:
            new_tokens.append(acc2[acc_count_all])
            acc_count_all += 1
    return new_tokens

def generate_efficient_feat_dicts_llama(wordseqs, tokenizer, lookback1, lookback2):
    text_dict = {}
    text_dict2 = {}
    text_dict3 = {}
    for es, story in enumerate(wordseqs.keys()):
        ds = wordseqs[story]
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token #[1 : <s> (start) token]
        acc8 = 0 # Number of word boundaries (i.e. 8 tokens appended to the end of the word)
        acc_words = 0
        print([tokenizer.decode(torch.tensor([i])) for i in [1, 29947, 29871]]) #<s>, 8, (space) (space : aka _ : a special token that is (when decoded as text) just a space)
        #=====before annotating #! remove later=======#
        decoded_before  = tokenizer.decode(torch.tensor(tokens))
        #=====remove above later!=======#
        for ei,i in enumerate(tokens):
            #i.e. if token starts with "_" (i.e. first subtoken of a word) AND if the token is not just a space
            # Add prefix number 8 before the token
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1): #i.e. not the last token
                # i.e. token itself is space AND next token is not the start of a new word 
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'): 
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                #i.e. it is in the middle of a word
                else:
                    acc.append(i)
            else: # i.e. last token
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data: #ds.data : list of actual words
            if i.strip() != '': #i.e. if it is not just a space, add to the count of words
                acc_words += 1
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947) #add 8 to the end of the last word (because the last word is not annotated with 8)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            #i.e. loop over words
            if w.strip() != '' and w != "'s": #i.e. not just space NOR 's (ex : there's) 
                #i.e. window < 256 or (256 <= window < 512)
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    text_dict3[tuple(new_tokens)] = False
                #if window becomes 512 (boundary)
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    text_dict3[tuple(new_tokens)] = False
                else:
                    assert False
            else: #if it is sapce or 's
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                text_dict3[tuple(new_tokens)] = False
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
    return text_dict, text_dict2, text_dict3

def convert_to_feature_mats_llama(wordseqs, tokenizer, lookback1, lookback2, text_dict3):
    text_dict = {}
    text_dict2 = {}
    featureseqs = {}
    for es, story in enumerate(wordseqs.keys()):
        #print(story)
        ds = wordseqs[story]
        newdata = []
        total_len = len(ds.data)
        text = [" ".join(ds.data)]
        inputs = tokenizer(text, return_tensors="pt")
        tokens = np.array(inputs['input_ids'][0])
        assert (29947 not in tokens) # Use a dummy token '8' for marking word cutoffs
        acc = [1] # Contexts should start with special START token
        acc8 = 0
        acc_words = 0
        for ei,i in enumerate(tokens):
            if tokenizer.convert_ids_to_tokens(torch.tensor([i]))[0][0] == '▁'  and (tokenizer.decode(torch.tensor([i])).strip() != ''):
                acc.append(29947)
                acc.append(i)
                acc8 += 1
            elif ei != (len(tokens) - 1):
                if (i == 29871) and (tokenizer.convert_ids_to_tokens(torch.tensor([tokens[ei+1]]))[0][0] != '▁'):
                    acc.append(29947)
                    acc.append(i)
                    acc8 += 1
                else:
                    acc.append(i)
            else:
                acc.append(i)
        decoded = tokenizer.decode(torch.tensor(acc))
        acc_words = 0
        for i in ds.data:
            if i.strip() != '':
                acc_words += 1
        #print(acc8, acc_words, story, es)
        assert acc8 == acc_words # Number of annotations should equal number of words
        acc.append(29947)
        acc_lookback = 0
        misc_offset = 0
        new_tokens = [1]
        for i, w in enumerate(ds.data):
            if w.strip() != '' and w != "'s":
                if acc_lookback < lookback1 or (lookback2 > acc_lookback and acc_lookback >= lookback1):
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = False
                    newdata.append(text_dict3[tuple(new_tokens)])
                elif acc_lookback == lookback2:
                    new_tokens = compute_correct_tokens_llama(acc, acc_lookback, i + misc_offset, total_len)
                    acc_lookback = lookback1
                    text_dict[(story, i)] = new_tokens
                    text_dict2[(story, i)] = True
                    newdata.append(text_dict3[tuple(new_tokens)])
                else:
                    assert False
            else:
                text_dict[(story, i)] = new_tokens
                text_dict2[(story, i)] = True
                newdata.append(text_dict3[tuple(new_tokens)])
                acc_lookback += 1
                misc_offset -= 1
                continue
            acc_lookback += 1
            if i == total_len - 1:
                text_dict2[(story, i)] = True
        featureseqs[story] = DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)
        
    import pdb ; pdb.set_trace()
    #! if not resample, then return featureseqs!! (unresampled, with .data and .data_times and .tr_times)
    downsampled_featureseqs = {}
    for story in featureseqs:
        downsampled_featureseqs[story] = featureseqs[story].chunksums('lanczos', window=3)
    return downsampled_featureseqs
