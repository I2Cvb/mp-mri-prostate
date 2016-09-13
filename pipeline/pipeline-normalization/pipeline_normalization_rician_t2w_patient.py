"""
The pipeline is intended to find the normalization parameters
for each patient to perform a rician normalization
"""

import os

from joblib import Parallel, delayed

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import RicianNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_t2w = 'T2W'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']

# Define a function to make parallel processing
def find_normalization_params(pat_t2w, pat_gt, label):
    # Create the normalization object and load the model
    t2w_norm = RicianNormalization(T2WModality())

    # Read the T2W
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(pat_t2w)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label, pat_gt)

    # Find the normalization parameters
    t2w_norm.fit(t2w_mod, ground_truth=gt_mod, cat=label[0])

    return t2w_norm


# Generate the different path to be later treated
path_patients_list_t2w = []
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the T2W data
    path_patients_list_t2w.append(os.path.join(path_patients, id_patient,
                                               path_t2w))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

t2w_norm_list = Parallel(n_jobs=48)(delayed(find_normalization_params)(
    p_t2w,
    p_gt,
    label_gt)
                                   for p_t2w, p_gt in
                                   zip(path_patients_list_t2w,
                                       path_patients_list_gt))

# Store the different normalization object
path_store = '/data/prostate/pre-processing/mp-mri-prostate/rician-t2w'
for t2w_obj, pat in zip(t2w_norm_list, id_patient_list):
    # Create the path where to store the data
    pat_chg = pat.lower().replace(' ', '_') + '_norm.p'
    filename = os.path.join(path_store, pat_chg)
    t2w_obj.save_to_pickles(filename)
