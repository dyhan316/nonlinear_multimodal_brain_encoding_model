#class that takes in which dataset to use and returns the dataset (ex : MNIST, CIFAR10, etc.)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold

import sys, os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import time 
import config
##################################################
################# BS,CV,Default ##################
##################################################
#for comparison : the idices for CV were :
# valid_indices : [    4     7    18 ... 27444 27447 27448]
# valid_indices : [    1     3     8 ... 27436 27438 27446]
# [    0     2     5 ... 27441 27443 27445]
#[   11    21    23 ... 27417 27422 27427]
# [   10    13    19 ... 27433 27440 27442]

class TrainValSplitIterator:
    r""" 
    TODO add : 
    0. option to modify whether train/val split based on stories or data points themselves. 
    
    ==k means different things for diffferent methods!!==
    * k in BS : number of bootstraps (i.e. number of iterations)
    * k in CV : number of folds (and as consequene, also the number of iterations)
    * k in default : propotin of validation set (i.e. 1/k)
    
    # Default k=5 because: 1/5 = 0.2 (i.e. 20% validation set)
    
    """
    def __init__(self, indices, method="default", k=5, iterator_seed=None):
        self.indices = indices
        self.method = method
        self.k = k
        self.iterator_seed = iterator_seed
        self.current_fold = 0
        self.splits = self._initialize_splits() #list of tuples (train_indices, valid_indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= len(self.splits):
            raise StopIteration
        
        result = self.splits[self.current_fold]
        self.current_fold += 1
        return result

    def _initialize_splits(self):
        if self.method == "CV":
            return self._generate_cv_indices()
        elif self.method == "StratifiedCV":
            raise NotImplementedError("Stratified CV is not implemented yet")
        elif self.method == "BS":
            raise NotImplementedError("Bootstrap is not implemented yet")
        elif self.method in ["default", "default_no_shuffle"]:
            return [self._generate_default_indices()]
        else:
            raise NotImplementedError(f"method {self.method} is not implemented yet")

    def _generate_cv_indices(self):
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.iterator_seed) if self.iterator_seed is not None else KFold(n_splits=self.k, shuffle=False)
        return list(kf.split(self.indices))

    def _generate_default_indices(self):
        if self.iterator_seed is not None : 
            np.random.seed(self.iterator_seed)
            np.random.shuffle(self.indices)
        
        val_size = len(self.indices) // self.k
        train_indices = self.indices[:-val_size]
        valid_indices = self.indices[-val_size:]
        return (train_indices, valid_indices)

##################################################
#################### DATASET #####################
##################################################

def get_dataset(dataset_type, X_dict, y_dict):
    r""" 
    WILL ALWAYS WORK UNDER THE ASSUMPTION THAT X, Y ARE DICTIONARY!! (with story keys and tensor values )
    * dataset_type : str, type of dataset to use 
        * plain : just stack the stories (when story data can be mixed without issues (e.g., MLP))
    # TODO: also add, make it flexible (for RNN etc.)(central command from DNN_main.py)
    """
    if dataset_type == "plain":
        return plain_dataset(X_dict, y_dict)
    elif dataset_type == "down_sample_fixed_window_vec_pred" : 
        if config.S_SETTINGS_2 : 
            assert config.S_SETTINGS['sync_len'] == config.S_SETTINGS_2['sync_len'], "sync_len should be the same for both modalities (coded assuming that this is true)"
        return VecPredFromSeqDataset(X_dict, y_dict, delay = config.E_SETTINGS['delay'], sync_len = config.S_SETTINGS['sync_len'],
                                     X_input_window = config.E_SETTINGS['rstim_window_size'],
                                     pad_max_len=None, padding_side=None, PAD=None, SEP=None #cuz fixed window size
                                     )
                                    # Warning: 1. delay is assumed to be same for both stimuli (valid assumption I think) 
                                    #2. sync len is fixed to that of the first (i.e. config.S_SETTINGS) cuz it is assumed the same for both modalities    
    elif dataset_type == "down_sample_fixed_window" : 
        raise NotImplementedError("not verified...")
        return dualseq_dataset(X_dict, y_dict,
                               X_input_window = config.E_SETTINGS['rstim_window_size'], X_output_window = 1, 
                               Y_input_window = config.E_SETTINGS['resp_window_size'] , Y_output_window = 1,)
    elif dataset_type == "down_sample_fixed_window_stim_only":
        #TODO : figure out how to handle Y_input_window... 1??? (since it doesn't matter) or sth else??
        raise NotImplementedError("this has to be implemented!! especially how to handle resp_window_size and multi-layer approaches")
        return dualseq_dataset_single(X_dict, y_dict, 
                                      X_input_window = config.E_SETTINGS['rstim_window_size'], X_output_window = 1, 
                                      Y_input_window = config.E_SETTINGS['resp_window_size'] , Y_output_window = 1,
                                      return_what='X')
    elif dataset_type == "down_sample_fixed_window_resp_only":
        return dualseq_dataset_single(X_dict, y_dict, 
                                      X_input_window = config.E_SETTINGS['rstim_window_size'], X_output_window = 1, 
                                      Y_input_window = config.E_SETTINGS['resp_window_size'] , Y_output_window = 1,
                                      return_what='Y')
    else : 
        raise NotImplementedError(f"dataset_type {dataset_type} is not implemented yet")
    

def stack_data(data_dict, stories):
    r""" 
    * stacks the data (i.e. dict input X or y), given the stories 
    * stories : intentially an input, to prevent cases where the story sequences are mixed up 
    * probably used for cases where multiple stories can be mixed
    """
    assert len(data_dict) == len(stories), f"len of data_dict {len(data_dict)} should be equal to len of stories {len(stories)}"
    return np.vstack([data_dict[story] for story in stories])

class plain_dataset_mem_inefficient(Dataset):
    r"""  
    * dataset where the stories are just stacked 
    """
    def __init__(self, X_dict, y_dict):
        self.stories = list(X_dict.keys()) #fix the order of stories
        self.X = stack_data(X_dict, self.stories)
        self.y = stack_data(y_dict, self.stories)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx): #change to float32!!
        return self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)

class plain_dataset_mem_efficient(Dataset):
    r"""  
    * dataset where the stories are just stacked 
    """
    def __init__(self, X_dict, y_dict):
        self.stories = list(X_dict.keys()) #fix the order of stories
        self.rand_int = int(time.time())
        self.X_dir = os.path.join(config.DUMP_DIR, f'X_{self.rand_int}.npy')
        self.y_dir = os.path.join(config.DUMP_DIR, f'y_{self.rand_int}.npy')
        
        #make sure no accidental name clash
        assert not os.path.exists(self.X_dir), f"{self.X_dir} already exists."
        assert not os.path.exists(self.y_dir), f"{self.y_dir} already exists."
        
        np.save(self.X_dir, stack_data(X_dict, self.stories))
        np.save(self.y_dir, stack_data(y_dict, self.stories))
        
    def __len__(self):
        return np.load(self.X_dir, mmap_mode = 'r').shape[0]
    
    def __getitem__(self, idx): #change to float32!!
        return np.load(self.X_dir, mmap_mode = 'r')[idx].astype(np.float32), np.load(self.y_dir, mmap_mode = 'r')[idx].astype(np.float32)
    
class plain_dataset:
    """ Main dataset class that initializes as memory-efficient or memory-inefficient. """
    def __init__(self, X_dict, y_dict, mem_efficient=False):
        if mem_efficient:
            self.dataset = plain_dataset_mem_efficient(X_dict, y_dict)
        else:
            self.dataset = plain_dataset_mem_inefficient(X_dict, y_dict)
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

        

class ConcatDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # Perform the label equality check only once during initialization
        self._assert_labels_equal()

    def _assert_labels_equal(self):
        for idx in range(len(self.dataset1)):
            _, label1 = self.dataset1[idx]
            _, label2 = self.dataset2[idx]
            # Use numpy array equality check
            assert np.array_equal(label1, label2), f"Labels at index {idx} do not match."

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        # Assuming the first element is the feature and the second is the label
        feature1, label1 = self.dataset1[idx]
        feature2, _ = self.dataset2[idx]
        
        # Combine the features from both datasets
        combined_feature = np.concatenate((feature1, feature2), axis=-1) #! -1 cuz if if it's time series, then it's (seq_len, dim) (for dataset index)
        
        return combined_feature.astype(np.float32), label1.astype(np.float32)
    

class CombineDataset(ConcatDataset):
    def __init__(self, dataset1, dataset2):
        super().__init__(dataset1, dataset2)
        
    def __getitem__(self, idx):
        # Assuming the first element is the feature and the second is the label
        feature1, label1 = self.dataset1[idx]
        feature2, _ = self.dataset2[idx]
        
        return (feature1.astype(np.float32), feature2.astype(np.float32)), label1.astype(np.float32)

#! change below!!
"""
class CombineSeparatelyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2), "Datasets must be of the same length"
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        # Assuming the first element is the feature and the second is the label
        feature1, label1 = self.dataset1[idx]
        feature2, label2 = self.dataset2[idx]
        
        return (feature1, feature2), (label1, label2)
"""

class dualseq_dataset(Dataset):
    r""" 
    dataset for dual seoquence dataset, 
    * either provide input window or max_len, not both!
    * X_dict : dict of input sequences of shape (N + sync_len,D)
    * Y_dict : dict of output sequences of shape (N,D)
    * delay : 1 가 default (i.e. predict the "next" token (or vector) ) (i.e. past TR을 몇개전까지 볼건지)
    * sync_len : the difference between X_dict and Y_dict (10 in the case!) (sync from front!! ) 
        * if sync_len > 0 : X is longer than Y, if sync_len < 0 : Y is longer than X
    
    __getitem__ returns (X, Y, X_shifted, Y_shifted) #X_shifted의 마지막은 안씀, Y_shifted의 마지막은 next token prediction용으로 쓰일듯
    (shape of X : (input_window, D), shape of Y : (output_window, D)) (batchify and sending seq_len first and so on==> done in dataloader (and collate_fn))
    """
    def __init__(self, X_dict, Y_dict, delay=1, sync_len = 10, 
                 X_input_window = None, X_output_window = None, Y_input_window = None , Y_output_window = None, #* fixed window sizes 일때
                 pad_max_len = None, padding_side = None, PAD=None, #* variable window sizes 일때 (i.e. hence need padding ) (PAD : pad token)
                 SEP = None, #SEP : SEP token
                 ):
        self.stories = list(X_dict.keys())
        self.delay, self.sync_len = delay, sync_len
        self.X_input_window, self.X_output_window, self.Y_input_window, self.Y_output_window = X_input_window, X_output_window, Y_input_window, Y_output_window
        self.pad_max_len, self.padding_side, self.PAD = pad_max_len, padding_side, PAD
        self.SEP = SEP

        assert (X_input_window is not None and X_output_window is not None and Y_input_window is not None and Y_output_window is not None) or (pad_max_len is not None), "either provide fixed window or pad_max_len, not both!"
        
        if self.X_output_window!=1 or self.Y_output_window!=1 :
            raise NotImplementedError("output_window!=1 not checked yet... make sure it works properly (but probably would not be used here)")
        if self.delay != 1 :
            raise NotImplementedError("delay!=1 not checked yet... make sure it works properly")
        
        X_input_list, X_output_list = [], []
        Y_input_list, Y_output_list = [], []
        
        for story in self.stories :
            X_story = X_dict[story]
            Y_story = Y_dict[story]
            assert X_story.shape[0] == Y_story.shape[0] + sync_len, f"X_story.shape[0] {X_story.shape[0]} should be equal to Y_story.shape[0] {Y_story.shape[0]} + sync_len {sync_len}"
            
            if pad_max_len :
                assert padding_side and PAD, "padding_side and PAD must be provided if pad_max_len is provided"
                raise NotImplementedError("padding not implemented yet")
            if SEP :
                raise NotImplementedError("SEP not implemented yet")
            
            X_input, X_output = create_inout_sequences(X_story, self.X_input_window, self.X_output_window)
            Y_input, Y_output = create_inout_sequences(Y_story, self.Y_input_window, self.Y_output_window)
            X_input, X_output, Y_input, Y_output = self.realign_X_Y(X_input, X_output, Y_input, Y_output)
            
            X_input_list.append(X_input) ; X_output_list.append(X_output)
            Y_input_list.append(Y_input) ; Y_output_list.append(Y_output)
        
        self.stacked_X_input = np.vstack(X_input_list)
        self.stacked_X_output = np.vstack(X_output_list)
        self.stacked_Y_input = np.vstack(Y_input_list)
        self.stacked_Y_output = np.vstack(Y_output_list)
        
    def __len__(self) :
        return self.stacked_X_input.shape[0]
    
    def __getitem__(self, index) :
        return torch.from_numpy(self.stacked_X_input[index]).float(), torch.from_numpy(self.stacked_X_output[index]).float(), torch.from_numpy(self.stacked_Y_input[index]).float(), torch.from_numpy(self.stacked_Y_output[index]).float()
        
    def realign_X_Y(self, X_input, X_output, Y_input, Y_output) :
        r"""  
        sync_len, X_input_window, Y_input_window 를 고려해서 X또는 Y 초반을 제거하기! (so that the end of X and Y aligns)
        #i.e. realign WRT end (by removing front of seq)
        #!주의 : sync_len>0, output_window size가 X,Y둘다 1일때만으로 테스트함! 다른때는 모르겠음!
        """
        if self.sync_len >= 0 : #i.e. X comes before Y (X is longer)
            pseudo_start_idx = self.sync_len - (self.X_input_window - self.Y_input_window)
            if pseudo_start_idx >= 0 : 
                X_input = X_input[pseudo_start_idx :]
                X_output = X_output[pseudo_start_idx:]
            elif pseudo_start_idx < 0 :
                Y_input = Y_input[-pseudo_start_idx :]
                Y_output = Y_output[-pseudo_start_idx:]
        elif self.sync_len < 0 :
            #* 만들꺼면 그냥 X,Y뒤집고 sync_len뒤집고 똑같이 해주면 될듯
            raise NotImplementedError("sync_len < 0 not checked yet... make sure it works properly")
        return X_input, X_output, Y_input, Y_output
    
def create_inout_sequences(input_data, input_window ,output_window):
    assert output_window == 1 , "output_window!=1 not checked yet... make sure it works properly"
    L = len(input_data)
    block_num =  L - (input_window + output_window) + 1
    # total of [N - block_len + 1] blocks #i.e. num of shifted windows
    # where block_len = input_window + output_window

    total_input_seq = []
    total_output_seq = []
    for i in range( block_num ):
        input_seq = input_data[i : i + input_window] #1씩 shfited window하며 input_window만큼의 데이터를 가져옴
        output_seq = input_data[i + output_window : i + input_window + output_window]
        total_input_seq.append(input_seq)
        total_output_seq.append(output_seq)
    return np.array(total_input_seq), np.array(total_output_seq)
    

class dualseq_dataset_single(dualseq_dataset) :
    r""" 
    for single input/output (i.e. either X or Y) (inherits from dualseq_dataset)
    """
    def __init__(self, X_dict, Y_dict, delay=1, sync_len = 10, 
                 X_input_window = None, X_output_window = None, Y_input_window = None , Y_output_window = None, #* fixed window sizes 일때
                 pad_max_len = None, padding_side = None, PAD=None, #* variable window sizes 일때 (i.e. hence need padding ) (PAD : pad token)
                 SEP = None, #SEP : SEP token
                 return_what = 'X'
                 ):
        super().__init__(X_dict, Y_dict, delay, sync_len, X_input_window, X_output_window, Y_input_window, Y_output_window, pad_max_len, padding_side, PAD, SEP)
        self.return_what = return_what
        
    def __getitem__(self, index) :
        if self.return_what == 'X' :
            return torch.from_numpy(self.stacked_X_input[index]).float(), torch.from_numpy(self.stacked_X_output[index]).float()
        elif self.return_what == 'Y' :
            return torch.from_numpy(self.stacked_Y_input[index]).float(), torch.from_numpy(self.stacked_Y_output[index]).float()
        else :
            raise ValueError("return_what must be either 'X' or 'Y'")

class VecPredFromSeqDataset(Dataset):
    def __init__(self, X_dict, Y_dict, delay=1, sync_len=10, 
                 X_input_window=None,  # not None if Fixed window size for input sequence (mutually exclusive with options below )
                 pad_max_len=None, padding_side=None, PAD=None,  # not None if Max length for variable window size (padding needed)
                 SEP=None):
        """
        Args:
        X_dict: dict of input sequences of shape (N + sync_len, D)
        Y_dict: dict of output sequences of shape (N, D)
        delay: Predict the next token (default is 1) => here, default is differernt from dualseq_dataset
        sync_len: Difference between X_dict and Y_dict lengths (sync from front)
        X_input_window: Fixed window size for input sequence (if provided, assuems fixed window size)
        pad_max_len: Max length for variable window size (padding needed)
        padding_side: Side to pad (if pad_max_len is provided)
        PAD: Padding token
        SEP: Separator token (not implemented yet... probably not needed)
        """
        self.stories = list(X_dict.keys())
        self.delay, self.sync_len = delay, sync_len
        self.X_input_window = X_input_window
        self.pad_max_len, self.padding_side, self.PAD = pad_max_len, padding_side, PAD
        self.SEP = SEP

        assert (X_input_window is not None) or (pad_max_len is not None), "Either provide fixed window or pad_max_len, not both!"
        
        X_input_list, Y_output_list = [], []
        
        for story in self.stories:
            X_story = X_dict[story]
            Y_story = Y_dict[story]
            assert X_story.shape[0] == Y_story.shape[0] + sync_len, f"X_story.shape[0] {X_story.shape[0]} should be equal to Y_story.shape[0] {Y_story.shape[0]} + sync_len {sync_len}"
            
            if pad_max_len:
                assert padding_side and PAD, "padding_side and PAD must be provided if pad_max_len is provided"
                raise NotImplementedError("padding not implemented yet")
            if SEP:
                raise NotImplementedError("SEP not implemented yet")          
            X_input, Y_output = self.get_input_and_output(X_story, Y_story, self.X_input_window, self.delay, self.sync_len)    
            X_input_list.append(X_input)
            Y_output_list.append(Y_output)
        print("adhoc, but 'cuda'로 무조건 보내도록 해놓음! (cpu로 하면너무 느림..)")
        self.stacked_X_input = torch.from_numpy(np.vstack(X_input_list)).float().to('cuda') #.half().to('cuda') #half precision OK cuz feature extraction was done using float 16 in the first place
        self.stacked_Y_output = torch.from_numpy(np.vstack(Y_output_list)).float().to('cuda') #.half().to('cuda')
        print("stacked_X_input.shape : ", self.stacked_X_input.shape, "stacked_Y_output.shape : ", self.stacked_Y_output.shape)
        
    def __len__(self):
        return self.stacked_X_input.shape[0]
    
    def __getitem__(self, index):
        # return self.stacked_X_input[index].float(), self.stacked_Y_output[index].float()
        return self.stacked_X_input[index], self.stacked_Y_output[index]

    def get_input_and_output(self, input_data, output_data, input_window, delay, sync_len):
        L = len(input_data)
        block_num = L - input_window + 1
        
        total_input_seq = []
        total_output_seq = []
        for i in range(block_num - max(delay,0)) :
            if (i - sync_len + input_window + delay -1) <0 :
                print(f"skipped {i} cuz of negative index (would've looked at the back)(nothing to worry about, seems to happens when sync_len > input_window)")
                continue
            input_seq = input_data[i: i + input_window]
            output_seq = output_data[i - sync_len + input_window + delay -1 ]
            total_input_seq.append(input_seq)
            total_output_seq.append(output_seq)
        return np.array(total_input_seq), np.array(total_output_seq)
    
##################################################
################## DATALOADER ####################
##################################################
def get_dataloader(dataloder_type, dataset, batch_size, shuffle, **kwargs):
    if dataloder_type == "plain":
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif dataloder_type == "batch_second": 
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=batch_second_collate_fn)
    else : 
        raise NotImplementedError(f"dataloder_type {dataloder_type} is not implemented yet")


def RNN_data_collate(batch):
    r"""
    * collate function for RNN (i.e. when the data is not just a tensor)
    ##TODO... fake sth.. I don't know... must make custom dataloader for RNN and Transformers?
    """
    X, y = zip(*batch)
    return X, y

def batch_second_collate_fn(data):
    """
    * assumes data is list of size "batch_size", with each element of list being a tuple of size 2 (if dualseq_dataset_single) or 4(if dualseq_dataset)
    * returns a tuple of size 2 (if dualseq_dataset_single) or 4(if dualseq_dataset) of shape (input_window, batch_size, dim)
    * (automatically refrains from doing permute if dim is 2 (i.e. no time dim probably))
    """
    batched_data_list = []
    for idx, batched_tuple in enumerate(zip(*data)) : #* zip(*data) : [(X1,Y1), (X2,Y2), ...] -> [(X1,X2,...), (Y1,Y2,...)] (len2 or 4 tuple을 같은 type끼리 묶기)
        batched_data = torch.stack(batched_tuple) # batched_tuple : tuple of len "batch_size"
        if len(batched_data.shape) == 2 : #i.e. probably not time series data (i.e. batchdim, single vec ) (with not timepoint thing)
            batched_data_list.append(batched_data)
        else :
            batched_data_list.append(batched_data.permute(1,0,2)) #permute from (batch_size, input_window, dim) to (input_window, batch_size, dim)
    return tuple(batched_data_list)
