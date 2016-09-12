#!/bin/bash

# Define the path to the executable
pathToBin='../../bin/./reg_dce';

# Define the path to the data
pathData='/work/le2i/gu5306le/prostate/experiments/';
# Define the path to the DCE
pathDCE='/DCE'

# Define the path where to save the data
pathDCESave='/DCE_intra_reg'

# Keep the directory of the script into a variable
script_dir=$(pwd)

patient_idx=0
# For all the patients
for patient in $pathData*/; do
    
    # Save the patient directory name
    patient_folder=$(basename "$patient")

    script_filename_core='/reg_dce_'
    script_filename="$script_dir$script_filename_core$patient_idx"

    touch $script_filename
    : > $script_filename
    printf "$pathToBin \"$pathData$patient_folder$pathDCE\" \"$pathData$patient_folder$pathDCESave\"" >> $script_filename
    chmod u+x $script_filename
    qsub -q batch -pe smp 8 $script_filename
    
    ((patient_idx++))
    
done
