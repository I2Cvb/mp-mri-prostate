"""
The pipeline is intended to find the normalization parameters
for each patient to perform a piecewise-linear normalization
"""

import os

from joblib import Parallel, delayed

from protoclass.data_management import ADCModality
from protoclass.data_management import GTModality

from protoclass.preprocessing import PiecewiseLinearNormalization

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
# Define the path of the modality to normalize
path_adc = 'ADC'
# Define the path of the ground for the prostate
path_gt = 'GT_inv/prostate'
# Define the label of the ground-truth which will be provided
label_gt = ['prostate']
# Define the filename for the model
pt_mdl = '/data/prostate/pre-processing/mp-mri-prostate/adc-model/piecewise_model.npy'

# Define a function to make parallel processing
def find_normalization_params(pat_adc, pat_gt, label, pat_model):
    # Create the normalization object and load the model
    adc_norm = PiecewiseLinearNormalization(ADCModality())
    adc_norm.load_model(pat_model)

    # Read the ADC
    adc_mod = ADCModality()
    adc_mod.read_data_from_path(pat_adc)

    # Read the GT
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label, pat_gt)

    # Find the normalization parameters
    adc_norm.fit(adc_mod, ground_truth=gt_mod, cat=label[0])

    return adc_norm


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

adc_norm_list = Parallel(n_jobs=4)(delayed(find_normalization_params)(p_adc,
                                                                      p_gt,
                                                                      label_gt,
                                                                      pt_mdl)
                                   for p_adc, p_gt in
                                   zip(path_patients_list_adc,
                                       path_patients_list_gt))

# Store the different normalization object
path_store = '/data/prostate/pre-processing/mp-mri-prostate/piecewise-linear-adc'
for adc_obj, pat in zip(adc_norm_list, id_patient_list):
    # Create the path where to store the data
    pat_chg = pat.lower().replace(' ', '_') + '_norm.p'
    filename = os.path.join(path_store, pat_chg)
    adc_obj.save_to_pickles(filename)
