import joblib
import numpy as np
import pandas as pd
import cortex
import matplotlib.cm



SHAPE2SUB = {
        81126 : "UTS01",
        94251 : "UTS02",
        95556 : "UTS03",
} #! hardcoded!!

def UT_name_to_name(UT_name):
    return f'S{UT_name[-1]}'

def score_paths_from_exp_info(df, exp_info, sub):
    return df[df['exp_info']==exp_info][f'{sub}_score_paths'].values[0]

def load_arr_from_score(score_path, measure) : 
    return joblib.load(score_path)[measure]

def load_sub_measure_from_score(df, exp_info, sub, measure) :
    return load_arr_from_score(score_paths_from_exp_info(df, exp_info, sub), measure)

def get_pycortex_vol(arr, filter  = True, **kwargs):
    """sub is automatically determined from the shape of the array"""
    UT_name = SHAPE2SUB[arr.shape[0]]
    return cortex.Volume(arr, UT_name, UT_name + '_auto', **kwargs)

def get_pycortex_vol_2D(arr1, arr2, **kwargs):
    """sub is automatically determined from the shape of the array"""
    UT_name = SHAPE2SUB[arr1.shape[0]]
    print(arr1.shape, arr1.min(), arr1.max(), arr1.mean(), arr1.std())  
    print(arr2.shape, arr2.min(), arr2.max(), arr2.mean(), arr2.std())
    return cortex.Volume2D(arr1, arr2, UT_name, UT_name + '_auto', **kwargs)


def cortex_flat(vol, height = 128, **kwargs):
    cortex.quickshow(vol, with_colorbar=True, linewidth=4, thick=1, 
                            with_curvature=True, with_rois=True, with_labels = True, height=height, **kwargs)
    
def filter_noisy_voxels(arr, cc_max_threshold = 0.35):
    """Filter out voxels with low ccmax values"""
    #get sub and their ccmax
    UT_name = SHAPE2SUB[arr.shape[0]]
    sub = UT_name_to_name(UT_name)
    cc_max_arr = np.load(f'./misc_data_for_anlaysis/full_cc_max_{sub}.npy') #! HARDCODED!
    #filter out noisy voxels (NaN)
    arr[cc_max_arr <= cc_max_threshold] = np.nan
    return arr

###------plotter helpers------
def zoom_to_roi(subject, roi, hem, margin=10.0):
    #example usage : use zoom_to_roi(subject, 'FEF', 'right') after cortex.quickshow! 
    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1

    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
                                                                nudge=True)
    sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

    xmin, ymin = roi_pts.min(0) - margin
    xmax, ymax = roi_pts.max(0) + margin
    plt.axis([xmin, xmax, ymin, ymax])
