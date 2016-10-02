"""
This pipeline is used to resave the data from lemaitre-2016-nov for faster
loading.
"""

import os

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

from protoclass.data_management import DCEModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import StandardTimeNormalization

from protoclass.extraction import EnhancementSignalExtraction

from protoclass.classification import Classify

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_dce = 'DCE_reg_bspline'
# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']
# Define the path to the normalization parameters
path_norm = '/data/prostate/pre-processing/lemaitre-2016-nov/norm-objects'
# Define the path to store the Tofts data
path_store = '/data/prostate/extraction/mp-mri-prostate/ese-dce'

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
    path_patients_list_gt.append([os.path.join(path_patients, id_patient, gt)
                                 for gt in path_gt])

# Load all the data once. Splitting into training and testing will be done at
# the cross-validation time
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Load the testing data that correspond to the index of the LOPO
    # Create the object for the DCE
    dce_mod = DCEModality()
    dce_mod.read_data_from_path(path_patients_list_dce[idx_pat])
    print 'Read the DCE data for the current patient ...'

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Load the approproate normalization object
    filename_norm = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                     '_norm.p')
    dce_norm = StandardTimeNormalization.load_from_pickles(
        os.path.join(path_norm, filename_norm))

    dce_mod = dce_norm.normalize(dce_mod)

    # Create the object to extrac data
    dce_ese = EnhancementSignalExtraction(DCEModality())

    # Concatenate the training data
    data = dce_ese.transform(dce_mod, gt_mod, label_gt[0])
    # Check that the path is existing
    if not os.path.exists(path_store):
        os.makedirs(path_store)
    pat_chg = (id_patient_list[idx_pat].lower().replace(' ', '_') +
               '_ese_' + '_dce.npy')
    filename = os.path.join(path_store, pat_chg)
    np.save(filename, data)
