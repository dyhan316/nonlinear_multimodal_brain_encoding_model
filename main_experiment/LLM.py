#wrapper and stuff for LLM embeddings!!

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer #bjkim

from torch.nn.functional import softmax

import config
from huggingface_hub import login
login("<PLZ PUT IN YOUR HUGGINGFACE TOKEN>") #! must be done 

class LLM():
    """wrapper for LLaMA model from Meta, available on Hugging Face
    """
    def __init__(self, path='placeholder', vocab=['placeholder'], device='cpu', llm_checkpoint = None, llm_tokenizer = None): 
        self.device = device
        print("LLM Loading...")
        ##DID : meta-llama/Meta-Llama-3-8B  meta-llama/Llama-2-70b-hf        
        self.model = AutoModelForCausalLM.from_pretrained(llm_checkpoint).eval().to(self.device) #.half() 
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer)
        
        #! TODO : DO AMP!! 
        #! try different model and tokenizers!
        #TODO : set maxlen or sth!
        #GPT2 versions!
        #https://jalammar.github.io/illustrated-gpt2/ 
        self.UNK_ID = 0

    def encode(self, words): #bjkim
        """Encode the words using the tokenizer to get corresponding IDs.
        
        Args:
            words (list of str): A list of words to encode.
        
        Returns:
            list of int: A list of token IDs corresponding to the input words.
        """
        #word to list method
        new_word = []

        for word in words:
            _, token_ids_list = encode_word(word, self.tokenizer)
            
            num_subtokens = len(token_ids_list)
            
            if num_subtokens > 1:
                new_word.extend(token_ids_list)
            elif num_subtokens == 0:
                new_word.append(self.UNK_ID)
            else: #num_subtokens == 1
                new_word.append(token_ids_list[0])
        
        return new_word

        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words) # this is array
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID

        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        
        return torch.tensor(story_array).long() # story x context

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    #original code, does batch-wise thing (I think)
    #! NOT USED!! AS TAKES TOO MUCH MEMORY!
    def get_hidden_batch_wise(self, ids, layer):
        """get hidden layer representations for LLaMA
        """
        mask = torch.ones(ids.shape).int()
        import pdb ; pdb.set_trace()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device), 
                                 attention_mask=mask.to(self.device), output_hidden_states=True)    
        return outputs.hidden_states[layer].detach().cpu().numpy() # Nearly n1 x n2 x n3 x n4 x n5 x n6 x n7 tensor...????
    
    
    #SINGLE BATCH!
    #! TODO : CHECK IF THIS IMPLEMENTATION IS ACTUALLY CORRECT!
    def get_hidden(self, ids, layer):
        """Get hidden layer representations for LLaMA with batch size 1 to minimize memory usage."""
        batch_size, seq_len = ids.shape
        all_hidden_states = []
        
        # Process each sequence individually
        for i in range(batch_size):
            single_id = ids[i].unsqueeze(0)  # Add batch dimension
            
            # Create an attention mask
            mask = torch.ones_like(single_id).int()
            
            with torch.no_grad():
                outputs = self.model(input_ids=single_id.to(self.device), 
                                    attention_mask=mask.to(self.device), 
                                    output_hidden_states=True)
                
            # Extract the hidden states at the specified layer
            hidden_states = outputs.hidden_states[layer]
            
            # Move the hidden states to the CPU and convert to numpy array
            hidden_states = hidden_states.detach().cpu().numpy()
            
            # Collect the hidden states
            all_hidden_states.append(hidden_states)
        
        # Concatenate the results along the batch dimension
        all_hidden_states = np.concatenate(all_hidden_states, axis=0)
        
        return all_hidden_states

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
        return probs

# @@ Static_ method
def encode_word(word,tokenizer): #bjkim
        """Encode the words using the tokenizer to get corresponding IDs.
        
        Args:
            words (list of str): A list of words to encode.
        
        Returns:
            list of int: A list of token IDs corresponding to the input words.
        """
        # print(words)
        token_list = []
        token_ids_list = []
        
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = [float(x) for x in ids]
        
        token_list.extend(tokens)  # Add the tokens for each word to the list
        token_ids_list.extend(ids)  # Add the token IDs for each word to the list
        # print(token_list)
        # print(token_ids_list)
        
        return token_list, token_ids_list