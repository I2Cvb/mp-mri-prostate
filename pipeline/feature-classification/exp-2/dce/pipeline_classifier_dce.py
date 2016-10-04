"""This pipeline is intended to make the classification of DCE modality
features."""
from __future__ import division

import os
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier

from protoclass.data_management import GTModality

# Define the path where the patients are stored
path_patients = '/data/prostate/experiments'
# Define the path where the features have been extracted
path_features = '/data/prostate/extraction/mp-mri-prostate'
# Define a list of the path where the feature are kept
dce_features = ['ese-dce']#,
#                'spatial-position-euclidean', 'spatial-dist-center',
#                'spatial-dist-contour']
# Define the extension of each features
ext_features = ['_ese__dce.npy']#, '_spe.npy', '_spe.npy',
#                '_spe.npy']
# Define the path of the balanced data
path_balanced = '/data/prostate/balanced/mp-mri-prostate/exp-2'
sub_folder = ['iht', 'nm1', 'nm2', 'nm3', 'rus', 'smote',
              'smote-b1', 'smote-b2', 'ros']
ext_balanced = '_dce.npz'
# Define the path of the ground for the prostate
path_gt = ['GT_inv/prostate', 'GT_inv/pz', 'GT_inv/cg', 'GT_inv/cap']
# Define the label of the ground-truth which will be provided
label_gt = ['prostate', 'pz', 'cg', 'cap']

# Generate the different path to be later treated
path_patients_list_gt = []
# Create the generator
id_patient_list = [name for name in os.listdir(path_patients)
                   if os.path.isdir(os.path.join(path_patients, name))]
id_patient_list = sorted(id_patient_list)

for id_patient in id_patient_list:
    # Append for the GT data - Note that we need a list of gt path
    path_patients_list_gt.append([os.path.join(path_patients, id_patient, gt)
                                  for gt in path_gt])

# Load all the data once. Splitting into training and testing will be done at
# the cross-validation time

data = []
data_bal = []
label = []
label_bal = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # For each patient we nee to load the different feature
    patient_data = []
    for idx_feat in range(len(dce_features)):
        # Create the path to the patient file
        filename_feature = (id_patient_list[idx_pat].lower().replace(' ', '_') +
                            ext_features[idx_feat])
        path_data = os.path.join(path_features, dce_features[idx_feat],
                                 filename_feature)
        single_feature_data = np.load(path_data)
        # Check if this is only one dimension data
        if len(single_feature_data.shape) == 1:
            single_feature_data = np.atleast_2d(single_feature_data).T
        patient_data.append(single_feature_data)

    # Concatenate the data in a single array
    patient_data = np.concatenate(patient_data, axis=1)

    print 'Imbalanced feature loaded ...'

    # Load the dataset from each balancing method
    data_bal_meth = []
    label_bal_meth = []
    for idx_imb in range(len(sub_folder)):
        path_bal = os.path.join(path_balanced, sub_folder[idx_imb])
        pat_chg = (id_patient_list[idx_pat].lower().replace(' ', '_') +
               '_dce.npz')
        filename = os.path.join(path_bal, pat_chg)
        npz_file = np.load(filename)
        data_bal_meth.append(npz_file['data_resampled'])
        label_bal_meth.append(npz_file['label_resampled'])

    data_bal.append(data_bal_meth)
    label_bal.append(label_bal_meth)

    print 'Balanced data loaded ...'

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Concatenate the training data
    data.append(patient_data)
    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'

results_bal = []
for idx_imb in range(len(sub_folder)):
    result_cv = []
    # Go for LOPO cross-validation
    for idx_lopo_cv in range(len(id_patient_list)):

        # Display some information about the LOPO-CV
        print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

        # Get the testing data
        testing_data = data[idx_lopo_cv]
        testing_label = label_binarize(label[idx_lopo_cv], [0, 255])
        print 'Create the testing set ...'

        # Create the training data and label
        # We need to take the balanced data
        training_data = [arr[idx_imb] for idx_arr, arr in enumerate(data_bal)
                         if idx_arr != idx_lopo_cv]
        training_label = [arr[idx_imb] for idx_arr, arr in enumerate(label_bal)
                         if idx_arr != idx_lopo_cv]
        # Concatenate the data
        training_data = np.vstack(training_data)
        training_label = label_binarize(np.hstack(training_label).astype(int),
                                        [0, 255])
        print 'Create the training set ...'

        # Perform the classification for the current cv and the
        # given configuration
        crf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        pred_prob = crf.fit(training_data,
                            np.ravel(training_label)).predict_proba(
                                testing_data)

        result_cv.append([pred_prob, crf.classes_])

    results_bal.append(result_cv)

# Save the information
path_store = '/data/prostate/results/mp-mri-prostate/exp-2/dce'
if not os.path.exists(path_store):
    os.makedirs(path_store)
joblib.dump(results_bal, os.path.join(path_store,
                                     'results.pkl'))
