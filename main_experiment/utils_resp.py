import os
import numpy as np
import h5py
import joblib
import config
import torch 

def get_resp(resp_setting, subject, stories, stack = True, vox = None, **kwargs):
    if "default" == resp_setting.split("_")[0]:
        return get_default_resp(subject, stories, stack, vox)
    elif "PCA" == resp_setting.split("_")[0]:
        vox_PCA_mat = vox #assume full vox!
        return get_PCA_resp(subject, stories, stack, vox, vox_PCA_mat, kwargs['PCA_N_COMP'])
    elif "PCAonIMP" == resp_setting.split("_")[0]:
        raise NotImplemented('not use this!!')
        #TODO : DO get_PCA_mat!! #DO DO DO DO DO (actually make it work without it!)
        assert len(stories) == 82, "PCAonIMP only implemented  for with all stories!"
        #vox : ignored here!!
        vox_to_use = get_PCA_mat(subject)['vox2keep']
        vox_PCA_mat = np.arange(vox_to_use.shape[0])
        #TODO : HERE, vox is ignored!! i.e. "vox"의 length가 PCA_N_COMP와 같다고 가정!!
        return get_PCA_resp(subject, stories, stack, vox_to_use, vox_PCA_mat ,kwargs['PCA_N_COMP'])
    else:
        raise NotImplementedError(f"setting {resp_setting} not implemented yet")

def get_default_resp(subject, stories, stack = True, vox = None):
    """loads response data
    """
    resp_all = joblib.load(os.path.join(config.SCALING_LAW_DATA_DIR, f"UT{subject[0]+'0'+subject[1]}_responses.jbl"))
    resp = {}
    for story in stories:
        resp[story] = np.nan_to_num(resp_all[story])
        if vox is not None:
            resp[story] = resp[story][:, vox]
    if stack: return np.vstack([resp[story] for story in stories])
    else: return resp

##PCA stuff 

# def get_IMP_data(subject): #NOT USE THIS!! use get_PCA_mat['vox2keep'] instead
#     print("주의 : IMP_data is assumed to be taken from ALL STORIES")
#     return np.load(file=f"{config.R_SETTINGS['IMP_VOXEL_DIR']}/{subject}_encoding_model_perceived.npz", allow_pickle=True)

def get_PCA_mat(subject):
    return np.load(file=f"{config.R_SETTINGS['PCA_MODEL']}/{subject}_PCA_8192.npz", allow_pickle=True)

def get_PCA_resp(subject, stories, stack = True, vox = None, vox_PCA_mat = None, n_comp = None): 
    org_data_all_stories = get_default_resp(subject, stories, stack = False) 
    """loads PCA responsedata, with n_comp"""
    PCA_mat = get_PCA_mat(subject)['pca_components']
    resp = {}
    #TODO : VOXEL ITSELF IS NOT GOOD!!!! THE IMP VOXELS ARE NOT USED!
    for story in stories:
        org_data = org_data_all_stories[story]
        if vox is not None : 
            # assert len(vox) == n_comp, f"# of vox {vox} should be equal to n_comp {n_comp}... if this error log, then you're probably using a wrong PCA weights or sth"
            org_data = org_data[:, vox] 
            resp[story] = org_data@((PCA_mat[:n_comp,vox_PCA_mat]).T) #embedded data
        else :
            resp[story] = org_data@((PCA_mat[:n_comp,:]).T) #embedded data
    if stack: return np.vstack([resp[story] for story in stories]) 
    else: return resp
    
def transform_data(original_data, subject, n_comp=None): #not actually used
    PCA_mat = get_PCA_mat(subject)['pca_components']
    return original_data@(PCA_mat[:n_comp,:]).T

def inverse_transform_data(PCA_embedding, subject, n_comp=None):
    PCA_mat = get_PCA_mat(subject)['pca_components']
    return PCA_embedding @ PCA_mat[:n_comp, :]

def get_test_resp(resp_setting, subject, experiment, task, **kwargs ):
    if "default" == resp_setting.split("_")[0]:
        hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject, experiment, task + ".hf5"), "r")
        resp = np.nan_to_num(hf["data"][:]) 
        hf.close()    
        return resp 
    elif "avg_test_resp" == resp_setting : #i.e. get the responses for the three stories... 
        test_stories = kwargs['stories']
        resp = get_resp("default", subject, test_stories, stack = False)        
        return resp
    elif "tensessions" == resp_setting.split("_")[0]:
        resp = joblib.load(os.path.join(config.DATA_TEST_DIR, "ten_test_response", f"tensessions_wheretheressmoke_{subject[:1]}0{subject[1:]}.jbl"))
        return resp    
    elif 'tensessionsPCA' == resp_setting.split("_")[0]:
        PCA_mat = get_PCA_mat(subject)['pca_components']
        org_data = joblib.load(os.path.join(config.DATA_TEST_DIR, "ten_test_response", f"tensessions_wheretheressmoke_{subject[:1]}0{subject[1:]}.jbl"))
        resp = org_data@((PCA_mat[:kwargs['PCA_N_COMP'],:]).T)
        return resp
    elif "PCA" == resp_setting.split("_")[0]:
        #by default set vox_to_use and vox_PCA_mat as None
        vox_to_use = None
        vox_PCA_mat = None
        return get_PCA_test_resp(subject, experiment, task, vox_to_use, vox_PCA_mat, kwargs['PCA_N_COMP'])
    elif "PCAonIMP" == resp_setting.split("_")[0]:
        vox_to_use = get_PCA_mat(subject)['vox2keep']
        vox_PCA_mat = np.arange(vox_to_use.shape[0])
        return get_PCA_test_resp(subject, experiment, task, vox_to_use, vox_PCA_mat, kwargs['PCA_N_COMP'])
    else:
        raise NotImplementedError(f"setting {resp_setting} not implemented yet")

def get_PCA_test_resp(subject, experiment, task, vox = None, vox_PCA_mat = None, n_comp = None ):
    """loads test (ex: wheretheresmoke) PCA responsedata, with n_comp, NO OPTION FOR VOX"""
    PCA_mat = get_PCA_mat(subject)['pca_components']
    
    hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject, experiment, task + ".hf5"), "r")
    org_data = np.nan_to_num(hf["data"][:])
    if vox is not None :
        # assert len(vox) == n_comp, f"# of vox {vox} should be equal to n_comp {n_comp}... if this error log, then you're probably using a wrong PCA weights or sth"
        org_data = org_data[:, vox]
        resp = org_data@((PCA_mat[:n_comp,vox_PCA_mat]).T) #embedded data    
    else : 
        resp = org_data@((PCA_mat[:n_comp,:]).T) #embedded data
    hf.close()
    return resp



    
# def get_resp(subject, stories, stack = True, vox = None):
#     """loads response data
#     """
#     subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
#     resp = {}
#     for story in stories:
#         resp_path = os.path.join(subject_dir, "%s.hf5" % story)
#         hf = h5py.File(resp_path, "r")
#         resp[story] = np.nan_to_num(hf["data"][:])
#         if vox is not None:
#             resp[story] = resp[story][:, vox]
#         hf.close()
#     if stack: return np.vstack([resp[story] for story in stories]) 
#     else: return resp
