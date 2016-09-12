#!/bin/bash

# Define the path to the executable
pathToBin='../../bin/./reg_adc';

# Define the path to the data
pathData='/work/le2i/gu5306le/prostate/experiments/';
# Define the path to the T2W GT
pathT2WGT='/GT_inv/prostate'
# Define the path to the ADC GT
pathADCGT='/adc_gt_prostate.nii.gz'
# Define the path to the ADC serie
pathADC='/ADC'

# Define the path where to save the data
pathADCSave='/ADC_reg_bspline'

# Keep the directory of the script into a variable
script_dir=$(pwd)

patient_idx=0
# For all the patients
for patient in $pathData*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    script_filename_core='/reg_adc_'
    script_filename="$script_dir$script_filename_core$patient_idx"

    touch $script_filename
    : > $script_filename
    printf "$pathToBin \"$pathData$patient_folder$pathT2WGT\" \"$pathData$patient_folder$pathADCGT\" \"$pathData$patient_folder$pathADC\" \"$pathData$patient_folder$pathADCSave\"" >> $script_filename
    chmod u+x $script_filename
    qsub -q batch -pe smp 8 $script_filename
    
    ((patient_idx++))
    
done
