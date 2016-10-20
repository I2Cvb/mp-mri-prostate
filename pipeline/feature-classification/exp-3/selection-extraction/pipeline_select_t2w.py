"""This pipeline is intended for selecting the more appropriate feature"""
import os
import numpy as np

from sklearn.externals import joblib

from collections import Counter

path_feat_sel = '/data/prostate/results/mp-mri-prostate/exp-3/selection-extraction/anova/t2w/feat_imp.pkl'
# Load the data and take the 25th percentile
feat_sel = joblib.load(path_feat_sel)[4]

lin_feat_sel = np.concatenate(feat_sel, axis=0)
count_feat =  Counter(lin_feat_sel)

# Get the most common element to get the correct size
feat_sel = count_feat.most_common(feat_sel[0].size)
idx_feat_val = np.array([elt[0] for elt in feat_sel])

# Save the information
path_store = '/data/prostate/results/mp-mri-prostate/exp-3/selection-extraction/anova/t2w'
if not os.path.exists(path_store):
    os.makedirs(path_store)
joblib.dump(idx_feat_val, os.path.join(path_store,
                                       'idx_sel_feat.pkl'))
