"""
This pipeline is used to find the parameters for each patient and
will be saved later.
"""

import os

import numpy as np

from joblib import Parallel, delayed

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_dce = 'DCE_reg_bspline'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the filename for the model
pt_mdl = '/data/prostate/pre-processing/lemaitre-2016-nov/model/model_stn.npy'

# Define a function to make parallel processing
def find_normalization_params(pat_dce, pat_gt, label, pat_model):
    # Create the normalization object and load the model
    dce_norm = StandardTimeNormalization(DCEModality())
    dce_norm.load_model(pat_model)

    # Read the DCE
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(pat_dce)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label, pat_gt)

    # Find the normalization parameters
    dce_norm.fit(dce_mod, ground_truth=gt_mod, cat=label[0])

    return dce_norm


# Generate the different path to be later treated
path_patients_list_dce = []
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the DCE data
    path_patients_list_dce.append(os.path.join(path_patients, id_patient,
                                               path_dce))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

dce_norm_list = Parallel(n_jobs=4)(delayed(find_normalization_params)(p_dce,
                                                                      p_gt,
                                                                      label_gt,
                                                                      pt_mdl)
                                   for p_dce, p_gt in
                                   zip(path_patients_list_dce,
                                       path_patients_list_gt))

# Store the different normalization object
path_store = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'
for dce_obj, pat in zip(dce_norm_list, id_patient_list):
    # Create the path where to store the data
    pat_chg = pat.lower().replace(' ', '_') + '_norm.p'
    filename = os.path.join(path_store, pat_chg)
    dce_obj.save_to_pickles(filename)
