"""
This pipeline is intended to extractedge information from T2W images.
"""

import os

import numpy as np

from protoclass.data_management import T2WModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import RicianNormalization
from protoclass.preprocessing import GaussianNormalization

from protoclass.extraction import EdgeSignalExtraction

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_t2w = 'T2W'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the path where the information for the gaussian normalization are
path_gaussian = '/data/prostate/pre-processing/mp-mri-prostate/gaussian-t2w'
# Define the path where the information for the rician normalization are
path_rician = '/data/prostate/pre-processing/mp-mri-prostate/rician-t2w'
# Define the path to store the Tofts data
path_store = '/data/prostate/extraction/mp-mri-prostate'

# ID of the patient for which we need to use the Gaussian Normalization
ID_GAUSSIAN = '387'

# Set the value of the extremum
EXTREM = (-4.48, 22.11)

# Set the value of the edges
TYPE_FILTER = ('sobel', 'prewitt', 'scharr', 'kirsch', 'laplacian')

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

# List where to store the different minimum
for id_p, (p_t2w, p_gt) in enumerate(zip(path_patients_list_t2w,
                                         path_patients_list_gt)):

    print 'Processing {}'.format(id_patient_list[id_p])

    # Remove a part of the string to have only the id
    nb_patient = id_patient_list[id_p].replace('Patient ', '')

    # Read the image data
    t2w_mod = T2WModality()
    t2w_mod.read_data_from_path(p_t2w)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, p_gt)

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

    # Rescale the data on 8 bits
    t2w_mod.data_ = ((t2w_mod.data_ - EXTREM[0]) *
                     (255. / (EXTREM[1] - EXTREM[0])))

    # Update the histogram
    t2w_mod.update_histogram()

    # Extract the edges for each type of filter and order of filter
    for type_f in TYPE_FILTER:
        print 'The {} will be extracted'.format(type_f)

        # Create the extraction method
        ext = EdgeSignalExtraction(t2w_mod, edge_detector=type_f)

        # Fit the data
        print 'Compute the edge map'
        ext.fit(t2w_mod, ground_truth=gt_mod, cat=label_gt[0])

        # Extract the data
        print 'Extract the edge map'
        data = ext.transform(t2w_mod, ground_truth=gt_mod, cat=label_gt[0])

        # Store the data
        print 'Store the data in the right directory'

        # Create the path for the current version of the filter
        path_filter = os.path.join(path_store, type_f)

        # Check that the path is existing
        if not os.path.exists(path_filter):
            os.makedirs(path_filter)
            pat_chg = (id_patient_list[id_p].lower().replace(' ', '_') +
                       '_edge_t2w.npy')
            filename = os.path.join(path_filter, pat_chg)
            np.save(filename, data)
