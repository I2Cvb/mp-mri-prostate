"""
This pipeline allow to find the minimum value of each volume to apply a common
offset
"""

import os

import numpy as np

from protoclass.data_management import T2WModality

from protoclass.preprocessing import RicianNormalization
from protoclass.preprocessing import GaussianNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_t2w = 'T2W'
# Define the path where the information for the gaussian normalization are
path_gaussian = '/data/prostate/pre-processing/mp-mri-prostate/gaussian-t2w'
# Define the path where the information for the rician normalization are
path_rician = '/data/prostate/pre-processing/mp-mri-prostate/rician-t2w'
# ID of the patient for which we need to use the Gaussian Normalization
ID_GAUSSIAN = '387'

# Generate the different path to be later treated
path_patients_list_t2w = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the T2W data
    path_patients_list_t2w.append(os.path.join(path_patients, id_patient,
                                               path_t2w))

# List where to store the different minimum
list_min = []
list_max = []
for id_p, p_t2w in enumerate(path_patients_list_t2w):

    # Remove a part of the string to have only the id
    nb_patient = id_patient_list[id_p].replace('Patient ', '')

    # Read the image data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(p_t2w)

    if not nb_patient == ID_GAUSSIAN:
        # Rician Normalization

        # Read the normalization information
        pat_chg = id_patient_list[id_p].lower().replace(' ', '_') + '_norm.p'
        filename = os.path.join(path_rician, pat_chg)
        t2w_norm = RicianNormalization.load_from_pickles(filename)

        # Normalize the data
        t2w_mod = t2w_norm.normalize(t2w_mod)

    else:
        # Gaussian Normalization

        # Read the normalization information
        pat_chg = id_patient_list[id_p].lower().replace(' ', '_') + '_norm.p'
        filename = os.path.join(path_gaussian, pat_chg)
        t2w_norm = GaussianNormalization.load_from_pickles(filename)

        # Normalize the data
        t2w_mod = t2w_norm.normalize(t2w_mod)

    # Append the minimum
    list_min.append(np.min(t2w_mod.data_))
    list_max.append(np.max(t2w_mod.data_))

print 'The min and max to consider for the whole dataset is {} - {}'.format(
    min(list_min), max(list_max))
