"""
This pipeline is intended to extract edge information from ADC images.
"""

import os

import numpy as np

from protoclass.data_management import ADCModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import PiecewiseLinearNormalization
from protoclass.extraction import EdgeSignalExtraction

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_adc = 'ADC_reg_bspline'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the path where the information for the piecewise-linear normalization
path_piecewise = '/data/prostate/pre-processing/mp-mri-prostate/piecewise-linear-adc'
# Define the path to store the Tofts data
path_store = '/data/prostate/extraction/mp-mri-prostate/edge-adc'

# Set the value of the edges
TYPE_FILTER = ('sobel', 'prewitt', 'scharr', 'kirsch', 'laplacian')

# Generate the different path to be later treated
path_patients_list_adc = []
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
for id_patient in id_patient_list:
    # Append for the ADC data
    path_patients_list_adc.append(os.path.join(path_patients, id_patient,
                                               path_adc))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

# List where to store the different minimum
for id_p, (p_adc, p_gt) in enumerate(zip(path_patients_list_adc,
                                         path_patients_list_gt)):

    print 'Processing {}'.format(id_patient_list[id_p])

    # Remove a part of the string to have only the id
    nb_patient = id_patient_list[id_p].replace('Patient ', '')

    # Read the image data
    adc_mod = ADCModality()
    adc_mod.read_data_from_path(p_adc)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, p_gt)

    # Read the normalization information
    pat_chg = id_patient_list[id_p].lower().replace(' ', '_') + '_norm.p'
    filename = os.path.join(path_piecewise, pat_chg)
    adc_norm = PiecewiseLinearNormalization.load_from_pickles(filename)

    # Normalize the data
    adc_mod = adc_norm.normalize(adc_mod)

    # Extract the edges for each type of filter and order of filter
    for type_f in TYPE_FILTER:
        print 'The {} will be extracted'.format(type_f)

        # Create the extraction method
        ext = EdgeSignalExtraction(adc_mod, edge_detector=type_f)

        # Fit the data
        print 'Compute the edge map'
        ext.fit(adc_mod, ground_truth=gt_mod, cat=label_gt[0])

        # Extract the data
        print 'Extract the edge map'
        data = ext.transform(adc_mod, ground_truth=gt_mod, cat=label_gt[0])

        # Store the data
        print 'Store the data in the right directory'

        # Create the path for the current version of the filter
        path_filter = os.path.join(path_store, type_f)

        # Check that the path is existing
        if not os.path.exists(path_filter):
            os.makedirs(path_filter)
        pat_chg = (id_patient_list[id_p].lower().replace(' ', '_') +
                   '_edge_adc.npy')
        filename = os.path.join(path_filter, pat_chg)
        np.save(filename, data)
