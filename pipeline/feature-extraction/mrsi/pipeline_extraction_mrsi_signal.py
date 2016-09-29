"""
This pipeline is intended to extract MRSI spectra with l2 normalization.
"""

import os

import numpy as np

from protoclass.data_management import RDAModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import MRSIPhaseCorrection
from protoclass.preprocessing import MRSIFrequencyCorrection
from protoclass.preprocessing import MRSIBaselineCorrection
from protoclass.extraction import MRSISpectraExtraction

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_mrsi = 'MRSI'
# Define the name of the MRSI
filename_mrsi = 'CSI_SE_3D_140ms_16c.rda'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the path to store the Tofts data
path_store = '/data/prostate/extraction/mp-mri-prostate/mrsi-spectra'

# Generate the different path to be later treated
path_patients_list_mrsi = []
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the MRSI data
    path_patients_list_mrsi.append(os.path.join(path_patients, id_patient,
                                               path_mrsi))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

# List where to store the different minimum
for id_p, (p_mrsi, p_gt) in enumerate(zip(path_patients_list_mrsi,
                                         path_patients_list_gt)):

    print 'Processing {}'.format(id_patient_list[id_p])

    # Remove a part of the string to have only the id
    nb_patient = id_patient_list[id_p].replace('Patient ', '')

    # Read the image data
    rda_mod = RDAModality(1250.)
    rda_mod.read_data_from_path(os.path.join(p_mrsi, filename_mrsi))

    # Correct the phase
    phase_correction = MRSIPhaseCorrection(rda_mod)
    rda_mod = phase_correction.transform(rda_mod)

    # Correct the frequency shift
    freq_correction = MRSIFrequencyCorrection(rda_mod)
    rda_mod = freq_correction.fit(rda_mod).transform(rda_mod)

    # Correct the baseline
    baseline_correction = MRSIBaselineCorrection(rda_mod)
    rda_mod = baseline_correction.fit(rda_mod).transform(rda_mod)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, p_gt)

    # Extract the data
    ext = MRSISpectraExtraction(rda_mod)
    data = ext.fit(rda_mod, gt_mod, label_gt[0]).transform(
        rda_mod, gt_mod, label_gt[0])

    # Check that the path is existing
    if not os.path.exists(path_store):
        os.makedirs(path_store)
    pat_chg = (id_patient_list[id_p].lower().replace(' ', '_') +
               '_spectra_mrsi.npy')
    filename = os.path.join(path_store, pat_chg)
    np.save(filename, data)
