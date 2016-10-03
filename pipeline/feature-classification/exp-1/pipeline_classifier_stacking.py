"""This pipeline is intended to make the classification of ALL modality
features."""
from __future__ import division

import os
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from protoclass.data_management import GTModality

# Define the path where the patients are stored
path_patients = '/data/prostate/experiments'
# Define the path where the features have been extracted
path_features = '/data/prostate/extraction/mp-mri-prostate'

# T2W
t2w_features = ['dct-t2w', 'edge-t2w/kirsch', 'edge-t2w/laplacian',
                'edge-t2w/prewitt', 'edge-t2w/scharr', 'edge-t2w/sobel',
                'gabor-t2w', 'harlick-t2w', 'ise-t2w', 'lbp-t2w', 'lbp-t2w',
                'phase-congruency-t2w']
t2w_ext_features = ['_dct_t2w.npy', '_edge_t2w.npy', '_edge_t2w.npy',
                    '_edge_t2w.npy', '_edge_t2w.npy', '_edge_t2w.npy',
                    '_gabor_t2w.npy', '_haralick_t2w.npy', '_ise_t2w.npy',
                    '_lbp_8_1_t2w.npy', '_lbp_16_2_t2w.npy',
                    '_phase_congruency_t2w.npy']
# ADC
adc_features = ['dct-adc', 'edge-adc/kirsch', 'edge-adc/laplacian',
                'edge-adc/prewitt', 'edge-adc/scharr', 'edge-adc/sobel',
                'gabor-adc', 'harlick-adc', 'ise-adc', 'lbp-adc', 'lbp-adc',
                'phase-congruency-adc']
adc_ext_features = ['_dct_adc.npy', '_edge_adc.npy', '_edge_adc.npy',
                    '_edge_adc.npy', '_edge_adc.npy', '_edge_adc.npy',
                    '_gabor_adc.npy', '_haralick_adc.npy', '_ise_adc.npy',
                    '_lbp_8_1_adc.npy', '_lbp_16_2_adc.npy',
                    '_phase_congruency_adc.npy']

# MRSI
mrsi_features = ['mrsi-spectra']
mrsi_ext_features = ['_spectra_mrsi.npy']

# DCE
dce_features = ['ese-dce']
dce_ext_features = ['_ese__dce.npy']

# Spatial information
spatial_features = ['spatial-position-euclidean', 'spatial-dist-center',
                    'spatial-dist-contour']
spatial_ext_features = ['_spe.npy', '_spe.npy',
                        '_spe.npy']

# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']

# Generate the different path to be later treated
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
# Sort the list of patient
id_patient_list = sorted(id_patient_list)

for id_patient in id_patient_list:
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient, gt)
                                  for gt in path_gt])

# Load all the data once. Splitting into training and testing will be done at
# the cross-validation time
data_t2w = []
data_adc = []
data_dce = []
data_mrsi = []
data_spatial = []
label = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Let's get the information about the pz
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_pz = gt_mod.extract_gt_data('pz', output_type='data')
    gt_pz = gt_pz[roi_prostate]

    # Read the T2W information
    patient_data_t2w = []
    for idx_feat in range(len(t2w_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            t2w_ext_features[idx_feat])
        path_data = os.path.join(path_features, t2w_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data_t2w.append(single_feature_data)

    # Concatenate the data
    patient_data_t2w = np.concatenate(patient_data_t2w, axis=0)
    data_t2w.append(patient_data_t2w)

    # Read the ADC information
    patient_data_adc = []
    for idx_feat in range(len(adc_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            adc_ext_features[idx_feat])
        path_data = os.path.join(path_features, adc_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data_adc.append(single_feature_data)

    # Concatenate the data
    patient_data_adc = np.concatenate(patient_data_adc, axis=0)
    data_adc.append(patient_data_adc)

    # Read the MRSI information
    patient_data_mrsi = []
    for idx_feat in range(len(mrsi_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            mrsi_ext_features[idx_feat])
        path_data = os.path.join(path_features, mrsi_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data_mrsi.append(single_feature_data)

    # Concatenate the data
    patient_data_mrsi = np.concatenate(patient_data_mrsi, axis=0)
    data_mrsi.append(patient_data_mrsi)

    # Read the DCE information
    patient_data_dce = []
    for idx_feat in range(len(dce_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            dce_ext_features[idx_feat])
        path_data = os.path.join(path_features, dce_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data_dce.append(single_feature_data)

    # Concatenate the data
    patient_data_dce = np.concatenate(patient_data_dce, axis=0)
    data_dce.append(patient_data_dce)

    # Read the SPATIAL information
    patient_data_spatial = []
    for idx_feat in range(len(spatial_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            spatial_ext_features[idx_feat])
        path_data = os.path.join(path_features, spatial_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data_spatial.append(single_feature_data)

    # Add the information about the pz
    patient_data_spatial.append(np.atleast_2d(gt_pz).T)
    # Concatenate the data
    patient_data_spatial = np.concatenate(patient_data_spatial, axis=0)
    data_spatial.append(patient_data_spatial)

    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'

# Create a list concatenating all the data
data = [data_t2w, data_adc, data_mrsi, data_dce, data_spatial]

result_cv = []
# Go for LOPO cross-validation
for idx_lopo_cv in range(len(id_patient_list)):

    # Display some information about the LOPO-CV
    print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

    # We will need a training and a validation set for the meta-classifier
    # Create a vector with all the patients
    idx_patient = range(len(id_patient_list))
    idx_patient.remove(idx_lopo_cv)
    idx_patient = np.roll(idx_patient, idx_lopo_cv)
    # We will use the 60 percent as training and 40 percent as validation
    idx_split = int(0.6 * (len(id_patient_list) - 1))
    idx_patient_training = idx_patient[:idx_split]
    idx_patient_validation = idx_patient[idx_split:]

    # Create an empty list for the ensemble of RF
    rf_ensemble = []

    # Get the label
    training_label = [arr for idx_arr, arr in enumerate(label)
                      if idx_arr in idx_patient_training]
    training_label = np.ravel(label_binarize(
        np.hstack(training_label).astype(int), [0, 255]))

    # We need to build the training and train each random forest
    for mod_data in range(len(data) - 1):
        # Get the training data
        # Create the training data and label
        training_data_mod = [arr for idx_arr, arr in enumerate(data[mod_data])
                             if idx_arr in idx_patient_training]
        # Get the spatial information
        training_data_spa = [arr for idx_arr, arr in enumerate(data[-1])
                             if idx_arr in idx_patient_training]
        # Concatenate the data
        training_data_mod = np.vstack(training_data_mod)
        training_data_spa = np.vstack(training_data_spa)
        # Concatenate spatial information and modality information
        training_data = np.hstack((training_data_mod, training_data_spa))

        # Create the current RF
        crf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        crf.fit(training_data, training_label)
        # Add the classifier
        rf_ensemble.append(crf)

    # Get the labels
    validation_label = [arr for idx_arr, arr in enumerate(label)
                        if idx_arr in idx_patient_validation]
    validation_label = np.ravel(label_binarize(
        np.hstack(validation_label).astype(int), [0, 255]))
    # We need to create the meta classifier
    rf_data_answer = []
    for mod_data in range(len(data) - 1):
        # Create the validation data and label
        validation_data_mod = [arr for idx_arr, arr
                               in enumerate(data[mod_data])
                               if idx_arr in idx_patient_validation]
        # Get the spatial information
        validation_data_spa = [arr for idx_arr, arr in enumerate(data[-1])
                               if idx_arr in idx_patient_validation]
        # Concatenate the data
        validation_data_mod = np.vstack(validation_data_mod)
        validation_data_spa = np.vstack(validation_data_spa)
        # Concatenate spatial information and modality information
        validation_data = np.hstack((validation_data_mod, validation_data_spa))

        # Get the validation through the already trained forest
        pred_proba = rf_ensemble[mod_data].predict_proba(validation_data)
        # Select only the column corresponding to the positive class
        pos_class_arg = np.ravel(np.argwhere(
            rf_ensemble[mod_data].classes_ == 1))[0]
        rf_data_answer.append(pred_proba[:, pos_class_arg])

    # For know we will train a classifier using the previous probability
    # extracted
    rf_data_answer = np.vstack(rf_data_answer).T

    # Create the meta-classifier
    cgb = GradientBoostingClassifier()
    cgb.fit(rf_data_answer, validation_label)

    testing_label = np.ravel(label_binarize(label[idx_lopo_cv], [0, 255]))
    # Go for the testing now
    testing_inter = []
    for mod_data in range(len(data) - 1):
        testing_data = data[mod_data][idx_lopo_cv]

        # Get the probability of the first layer
        pred_proba = rf_ensemble[mod_data].predict_proba(testing_data)
        # Select only the column corresponding to the positive class
        pos_class_arg = np.ravel(np.argwhere(
            rf_ensemble[mod_data].classes_ == 1))[0]
        testing_inter.append(pred_proba[:, pos_class_arg])

    # Make the classification with the second layer
    testing_inter = np.vstack(testing_inter).T
    pred_prob = cgb.predict_proba(testing_inter)
    result_cv.append([pred_prob, cgb.classes_])

# Save the information
path_store = '/data/prostate/results/mp-mri-prostate/exp-1/stacking'
if not os.path.exists(path_store):
    os.makedirs(path_store)
joblib.dump(result_cv, os.path.join(path_store,
                                    'results.pkl'))
