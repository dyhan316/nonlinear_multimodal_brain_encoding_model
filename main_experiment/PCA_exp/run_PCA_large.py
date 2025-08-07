##some preliminary

#getting all stories into a list 
stories = {"2": ["alternateithicatom", "souls", "avatar", "legacy", "odetostepfather"], "3": ["undertheinfluence", "howtodraw", "myfirstdaywiththeyankees", "naked", "life"], "4": ["stagefright", "tildeath", "sloth", "exorcism"], "5": ["haveyoumethimyet", "adollshouse", "inamoment", "theclosetthatateeverything", "adventuresinsayingyes", "buck"], "6": ["swimmingwithastronauts", "thatthingonmyarm", "eyespy", "itsabox", "hangtime"], "7": ["breakingupintheageofgoogle", "treasureisland", "shoppinginchina", "onlyonewaytofindout", "penpal"],  "8": ["goingthelibertyway", "kiksuya", "thepostmanalwayscalls", "backsideofthestorm", "sweetaspie", "thetiniestbouquet"], "9": ["lifeanddeathontheoregontrail", "thefreedomridersandme", "thumbsup", "becomingindian", "waitingtogo"], "10": ["singlewomanseekingmanwich", "whenmothersbullyback", "superheroesjustforeachother", "gpsformylostidentity", "catfishingstrangerstofindmyself", "christmas1940"], "11": ["stumblinginthedark", "forgettingfear", "bluehope", "lifereimagined", "ifthishaircouldtalk", "againstthewind"], "12": ["goldiethegoldfish", "quietfire", "cocoonoflove", "comingofageondeathrow", "leavingbaghdad"], "14": ["mybackseatviewofagreatromance", "reachingoutbetweenthebars", "afatherscover", "wildwomenanddancingqueens", "notontheusualtour", "food"], "15": ["vixenandtheussr", "golfclubbing", "beneaththemushroomcloud", "threemonths", "gangstersandcookies", "metsmagic"], "18": ["myfathershands", "theadvancedbeginner", "firetestforlove", "theshower", "birthofanation", "listo"], "20": ["thecurse", "thetriangleshirtwaistconnection", "mayorofthefreaks", "cautioneating", "theinterview", "jugglingandjesus"]}

all_stories = []
[all_stories.extend(story) for story in stories.values()]

print(all_stories)

import numpy as np
import h5py
import os
import argparse
import pdb
from sklearn.decomposition import PCA, IncrementalPCA #USE incrementalPCA, as it's faster and more memory efficient
from matplotlib import pyplot as plt

# Could have used get_resp function instead 
def concatenate_and_clean_datasets(source_dir, hf5_names, stack = True, vox = None):
    """loads response data
    """
    hf5_dir = os.path.join(source_dir)
    resp = {}
    for hf5_file in hf5_names:
        resp_path = os.path.join(hf5_dir, "%s.hf5" % hf5_file)
        hf = h5py.File(resp_path, "r")
        resp[hf5_file] = np.nan_to_num(hf["data"][:])
        if vox is not None:
            resp[hf5_file] = resp[hf5_file][:, vox]
        hf.close()
    if stack: return np.vstack([resp[story] for story in hf5_names]) 
    else: return resp
    
    
def apply_save_PCA(data, save_path, n_comp=None) : 
    r""" 
    * data : shape [n_samples, n_features]
    * save_path : path to save the pca result (npz that has PCA weight, explained_variance_ratio_, and pca_result)    
    """
    if n_comp is None :
        n_comp = min(data.shape[0], data.shape[1]) #take min as n_comp if not specified #NOT suggested, takes too long 
    pca = IncrementalPCA(n_components=n_comp, batch_size=n_comp)  # pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(data)
    np.savez(save_path, explained_variance_ratio=pca.explained_variance_ratio_, pca_components=pca.components_)
    
    
sub = "S1"
h5_folder_dir = f"/XXX/train_response/{sub}"

full_data = concatenate_and_clean_datasets(source_dir= h5_folder_dir ,
                                    hf5_names= all_stories) #all_stories[:5] # Check if 6 or 5 stories


apply_save_PCA(full_data, f'./PCA_exp/{sub}.npz', n_comp=5)
