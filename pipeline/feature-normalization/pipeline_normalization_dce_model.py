"""
This pipeline is used to compute the model which will be used for the
standard time normalization.
"""

import os

import numpy as np

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

# Generate the different path to be later treated
path_patients_list_dce = []
path_patients_list_gt = []
# Create the generator
id_patient_list = (name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name)))
for id_patient in id_patient_list:
    # Append for the DCE data
    path_patients_list_dce.append(os.path.join(path_patients, id_patient,
                                               path_dce))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

# Create the model iteratively
dce_norm = StandardTimeNormalization(DCEModality())
for pat_dce, pat_gt in zip(path_patients_list_dce, path_patients_list_gt):
    # Read the DCE
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(pat_dce)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, pat_gt)

    # Fit the model
    dce_norm.partial_fit_model(dce_mod, ground_truth=gt_mod,
                               cat=label_gt[0])

# Define the path where to store the model
path_store_model = '/data/prostate/pre-processing/lemaitre-2016-nov/model'
filename_model = os.path.join(path_store_model, 'model_stn.npy')

# Save the model
dce_norm.save_model(filename_model)
