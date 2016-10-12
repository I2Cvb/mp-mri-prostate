"""
This pipeline is used to report the results for the ADC modality.
"""

import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set(style='ticks', palette='Set2')
current_palette = sns.color_palette("deep", 7)

from scipy import interp

from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from protoclass.data_management import GTModality

from protoclass.validation import labels_to_sensitivity_specificity

# Define the path where all the patients are
path_patients = '/data/prostate/experiments'
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
label = []
for idx_pat in range(len(id_patient_list)):
    print 'Read patient {}'.format(id_patient_list[idx_pat])

    # Create the corresponding ground-truth
    gt_mod = GTModality()
    gt_mod.read_data_from_path(label_gt,
                               path_patients_list_gt[idx_pat])
    print 'Read the GT data for the current patient ...'

    # Extract the corresponding ground-truth for the testing data
    # Get the index corresponding to the ground-truth
    roi_prostate = gt_mod.extract_gt_data('prostate', output_type='index')
    # Get the label of the gt only for the prostate ROI
    gt_cap = gt_mod.extract_gt_data('cap', output_type='data')
    label.append(gt_cap[roi_prostate])
    print 'Data and label extracted for the current patient ...'

testing_label_cv = []
# Go for LOPO cross-validation
for idx_lopo_cv in range(len(id_patient_list)):

    # Display some information about the LOPO-CV
    print 'Round #{} of the LOPO-CV'.format(idx_lopo_cv + 1)

    testing_label = np.ravel(label_binarize(label[idx_lopo_cv], [0, 255]))
    testing_label_cv.append(testing_label)

fresults = '/data/prostate/results/mp-mri-prostate/exp-3/adc/k-best/results.pkl'
results = joblib.load(fresults)

# Select the best n% in ANOVA test
percentiles = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25])

# Create an handle for the figure
fig = plt.figure()
ax = fig.add_subplot(111)

for idx_p, p in enumerate(percentiles):
    # # Initialise a list for the sensitivity and specificity
    # Initilise the mean roc
    mean_tpr = []
    mean_fpr = np.linspace(0, 1, 30)
    auc_pat = []

    # Go for each cross-validation iteration
    for idx_cv in range(len(testing_label_cv)):

        # Print the information about the iteration in the cross-validation
        print 'Iteration #{} of the cross-validation'.format(idx_cv+1)

        # Get the prediction
        pred_score = results[idx_p][idx_cv][0]
        classes = results[idx_p][idx_cv][1]
        pos_class_arg = np.ravel(np.argwhere(classes == 1))[0]

        # Compute the fpr and tpr
        fpr, tpr, thresh = roc_curve(testing_label_cv[idx_cv],
                                     pred_score[:, pos_class_arg])

        # Compute the mean ROC
        mean_tpr.append(interp(mean_fpr,
                               fpr,
                               tpr))
        mean_tpr[idx_cv][0] = 0.0
        auc_pat.append(auc(mean_fpr, mean_tpr[-1]))

    avg_tpr = np.mean(mean_tpr, axis=0)
    std_tpr = np.std(mean_tpr, axis=0)
    avg_tpr[-1] = 1.0

    ax.plot(mean_fpr, avg_tpr,
            label=r'${}$ percentile - AUC $= {:1.3f} \pm {:1.3f}$'.format(
                p, auc(mean_fpr, avg_tpr), np.std(auc_pat)),
            lw=2)
    ax.fill_between(mean_fpr,
                    avg_tpr + std_tpr,
                    avg_tpr - std_tpr,
                    facecolor=current_palette[idx_p], alpha=0.2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title(r'ADC classification using the a subset of features using ANOVA f-score')

handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='lower right')#,
                #bbox_to_anchor=(1.4, 0.1))
# Save the plot
plt.savefig('results/exp-3/adc_anova.pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')
