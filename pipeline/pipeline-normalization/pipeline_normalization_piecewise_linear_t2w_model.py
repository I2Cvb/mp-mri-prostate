"""
This pipeline is attented to find the landmarks model to normalize
the T2W using the piecewise linear approach
"""

import os

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import PiecewiseLinearNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_t2w = 'T2W'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']

# Generate the different path to be later treated
path_patients_list_t2w = []
path_patients_list_gt = []
# Create the generator
id_patient_list = (name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name)))
for id_patient in id_patient_list:
    # Append for the T2W data
    path_patients_list_t2w.append(os.path.join(path_patients, id_patient,
                                               path_t2w))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

# Create the model iteratively
t2w_model = PiecewiseLinearNormalization(T2WModality())
for pat_t2w, pat_gt in zip(path_patients_list_t2w, path_patients_list_gt):
    # Read the T2W
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(pat_t2w)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, pat_gt)

    # Fit the model
    t2w_model.partial_fit_model(t2w_mod, ground_truth=gt_mod,
                                cat=label_gt[0])

# Define the path where to store the model
path_store_model = '/data/prostate/pre-processing/mp-mri-prostate/t2w-model'
filename_model = os.path.join(path_store_model, 'piecewise_model.npy')

# Save the model
t2w_model.save_model(filename_model)
