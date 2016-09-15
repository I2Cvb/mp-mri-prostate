"""
This pipeline is attented to find the landmarks model to normalize
the ADC using the piecewise linear approach
"""

import os

from protoclass.data_management import ADCModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import PiecewiseLinearNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_adc = 'ADC_reg_bspline'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']

# Generate the different path to be later treated
path_patients_list_adc = []
path_patients_list_gt = []
# Create the generator
id_patient_list = (name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name)))
for id_patient in id_patient_list:
    # Append for the ADC data
    path_patients_list_adc.append(os.path.join(path_patients, id_patient,
                                               path_adc))
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient,
                                               path_gt)])

# Create the model iteratively
adc_model = PiecewiseLinearNormalization(ADCModality())
for pat_adc, pat_gt in zip(path_patients_list_adc, path_patients_list_gt):
    # Read the ADC
    adc_mod = ADCModality()
    adc_mod.read_data_from_path(pat_adc)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt, pat_gt)

    # Fit the model
    adc_model.partial_fit_model(adc_mod, ground_truth=gt_mod,
                                cat=label_gt[0])

# Define the path where to store the model
path_store_model = '/data/prostate/pre-processing/mp-mri-prostate/adc-model'
filename_model = os.path.join(path_store_model, 'piecewise_model.npy')

# Save the model
adc_model.save_model(filename_model)
