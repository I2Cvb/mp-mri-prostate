Multi-parametric MRI for prostate cancer detection
==================================================

This pipeline is related to the chapter 6 of the following [PhD thesis](https://github.com/glemaitre/phd_thesis/blob/master/thesis.pdf)

How to use the pipeline?
------------------------

### Data registration

#### Code compilation

Before to go to data mining and mahcine learning, there is a need to register the DCE-MRI data and the ground-truth.
The registration was programed in C++ with the ITK toolbox. You need to compile the code to be able to call the executable.

Therefore, you can compile the code from the root directory as:

```
$ mkdir build
$ cd build
$ cmake ../src
```

Two executables will be created in `bin/`:

- `reg_dce`: register DCE-MRI data to remove motion during the acquisition.
- `reg_gt`: register the T2W-MRI, DCE-MRI, and ground-truth data.
- `reg_adc`: register the T2W-MRI, ADC, and ground-truth data.

#### Run the executables

You can call the executable `./reg_dce` as:

```
./reg_dce arg1 arg2
```

- `arg1` is the folder with the DCE-MRI data,
- `arg2` is the storage folder with the registered DCE-MRI data.

You can call the executable `./reg_gt` as:

```
./reg_gt arg1 arg2 arg3 arg4
```

- `arg1` is the T2W-MRI ground-truth with the segmentation of the prostate.
- `arg2` is the DCE-MRI ground-truth with the segmentation of the prostate.
- `arg3` is the folder with the DCE-MRI with intra-modality motion correction (see `reg_dce`),
- `arg4` is the storage folder inter-modality motion correction.

You can call the executable `./reg_adc` as:

```
./reg_gt arg1 arg2 arg3 arg4
```

- `arg1` is the T2W-MRI ground-truth with the segmentation of the prostate.
- `arg2` is the ADC ground-truth with the segmentation of the prostate.
- `arg3` is the folder with the ADC volume,
- `arg4` is the storage folder inter-modality motion correction.

### Normalization pipeline

The following normalization routines were applied:

- Rician normalization for T2W whenever possible and Gaussian normalization for one of the patient,
- Piecewise-linear normalization for the ADC data,
- Standard time normalization for the DCE data.

#### Run the pipeline

To normalize data, launch an `ipython` or `python` prompt and run from the root directory:

```python
>> run pipeline/feature-normalization/pipeline_normalization_rician_t2w_patient.py
>> run pipeline/feature-normalization/pipeline_normalization_gaussian_t2w_patient.py
>> run pipeline/feature-normalization/pipeline_normalization_piecewise_adc_model.py
>> run pipeline/feature-normalization/pipeline_normalization_piecewise_adc_patient.py
>> run pipeline/feature-normalization/pipeline_normalization_dce_model.py
>> run pipeline/feature-normalization/pipeline_normalization_dce_patient.py
```

### Extraction pipeline

The following extraction routines were applied:

- Intensity
- Edge
- Gabor
- Phase congruency
- Haralick
- DCT

#### Run the pipeline

To extract the different feature, launch an `ipython` or `python` prompt and run from the root directory:

##### T2W

```python
>> run pipeline/feature-extraction/t2w/pipeline_extraction_intensity_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_edge_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_haralick_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_phase_congruency_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_dct_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_gabor_t2w.py
```

##### ADC

```python
>> run pipeline/feature-extraction/adc/pipeline_extraction_intensity_adc.py
>> run pipeline/feature-extraction/adc/pipeline_extraction_edge_adc.py
>> run pipeline/feature-extraction/adc/pipeline_extraction_haralick_adc.py
>> run pipeline/feature-extraction/adc/pipeline_extraction_phase_congruency_adc.py
>> run pipeline/feature-extraction/adc/pipeline_extraction_dct_adc.py
>> run pipeline/feature-extraction/adc/pipeline_extraction_gabor_adc.py
```

##### Spatial

```python
>> run pipeline/feature-extraction/spatial/pipeline_extraction_distance_center.py
>> run pipeline/feature-extraction/spatial/pipeline_extraction_distance_contour.py
>> run pipeline/feature-extraction/spatial/pipeline_extraction_position_cylindrical.py
>> run pipeline/feature-extraction/spatial/pipeline_extraction_position_euclidean.py
```

##### MRSI

```python
>> run pipeline/feature-extraction/mrsi/pipeline_extraction_mrsi_signal.py
>> run pipeline/feature-extraction/mrsi/pipeline_extraction_relative_quantification.py
```

### Performing the different experiment

#### Experiment 1

```python
>> run pipeline/feature-classification/exp-1/pipeline_classifier_adc.py
>> run pipeline/feature-classification/exp-1/pipeline_classifier_dce.py
>> run pipeline/feature-classification/exp-1/pipeline_classifier_t2w.py
>> run pipeline/feature-classification/exp-1/mrsi/pipeline_classifier_mrsi_citrate_choline_fit.py
>> run pipeline/feature-classification/exp-1/mrsi/pipeline_classifier_mrsi_citrate_choline_fit_ratio.py
>> run pipeline/feature-classification/exp-1/mrsi/pipeline_classifier_mrsi_citrate_choline_no_fit.py
>> run pipeline/feature-classification/exp-1/mrsi/pipeline_classifier_mrsi_spectra.py
```

#### Experiment 2

```python
>> run pipeline/feature-classification/exp-2/pipeline_classifier_aggregation.py
>> run pipeline/feature-classification/exp-2/pipeline_classifier_stacking_adaboost.py
>> run pipeline/feature-classification/exp-2/pipeline_classifier_stacking_gradient_boosting.py
```

#### Experiment 3

##### Balancing prior to classification

The balancing is performed using:

- SMOTE
- SMOTE-b1
- SMOTE-b2
- NearMiss1
- NearMiss2
- NearMiss3
- Instance Hardness Threshold

First balancing should be performed as:

```python
>> run pipeline/feature-balancing/pipeline_balancing_adc.py
>> run pipeline/feature-balancing/pipeline_balancing_dce.py
>> run pipeline/feature-balancing/pipeline_balancing_t2w.py
>> run pipeline/feature-balancing/pipeline_balancing_mrsi.py
>> run pipeline/feature-balancing/pipeline_balancing_aggregation.py
```

Then, the classification is performed with:

```python
>> run pipeline/feature-classification/exp-3/balancing/pipeline_classifier_adc.py
>> run pipeline/feature-classification/exp-3/balancing/pipeline_classifier_dce.py
>> run pipeline/feature-classification/exp-3/balancing/pipeline_classifier_t2w.py
>> run pipeline/feature-classification/exp-3/balancing/pipeline_classifier_mrsi.py
>> run pipeline/feature-classification/exp-3/balancing/pipeline_classifier_aggregation.py
```

##### Selection/extraction with classification

The feature selection and classification are performed jointly.

Using ANOVA-based selection, run the following commands:

```python
>> run pipeline/feature-classification/exp-3/selection-extraction/anova/pipeline_classifier_adc.py
>> run pipeline/feature-classification/exp-3/selection-extraction/anova/pipeline_classifier_t2w.py
>> run pipeline/feature-classification/exp-3/selection-extraction/anova/pipeline_classifier_aggregation.py
```

Using Gini importance selection, run the following commands:

```python
>> run pipeline/feature-classification/exp-3/selection-extraction/rf/pipeline_classifier_adc.py
>> run pipeline/feature-classification/exp-3/selection-extraction/rf/pipeline_classifier_t2w.py
>> run pipeline/feature-classification/exp-3/selection-extraction/rf/pipeline_classifier_aggregation.py
```

The extraction is performed using:

- PCA
- Sparse-PCA
- ICA

Run the following commands:

```python
>> run pipeline/feature-classification/exp-3/selection-extraction/ica/pipeline_classifier_dce.py
>> run pipeline/feature-classification/exp-3/selection-extraction/ica/pipeline_classifier_mrsi.py
>> run pipeline/feature-classification/exp-3/selection-extraction/pca/pipeline_classifier_dce.py
>> run pipeline/feature-classification/exp-3/selection-extraction/pca/pipeline_classifier_mrsi.py
>> run pipeline/feature-classification/exp-3/selection-extraction/sparse-pca/pipeline_classifier_dce.py
>> run pipeline/feature-classification/exp-3/selection-extraction/sparse-pca/pipeline_classifier_mrsi.py
```

#### Experiment 4

To perform the fine-tuned classification, run the following commands:

```python
>> run pipeline/feature-classification/exp-4/pipeline_classifier_aggregation_modality.py
>> run pipeline/feature-classification/exp-4/pipeline_classifier_stacking.py
```

#### Experiment 5

To perform the fine-tuned classification without the MRSI data, run the following commands:

```python
>> run pipeline/feature-classification/exp-5/pipeline_classifier_stacking.py
```

### Plot

A set of plot can be generated for the different experiments:

```python
>> run pipeline/feature-validation/exp-1/pipeline_validation_mrsi.py
>> run pipeline/feature-validation/exp-1/pipeline_validation_t2w_adc.py
>> run pipeline/feature-validation/exp-2/pipeline_validation_coarse_combination.py
>> run pipeline/feature-validation/exp-3/balancing/pipeline_validation_adc.py
>> run pipeline/feature-validation/exp-3/balancing/pipeline_validation_dce.py
>> run pipeline/feature-validation/exp-3/balancing/pipeline_validation_mrsi.py
>> run pipeline/feature-validation/exp-3/balancing/pipeline_validation_t2w.py
>> run pipeline/feature-validation/exp-3/balancing/pipeline_validation_aggregation.py
>> run pipeline/feature-validation/exp-4/pipeline_validation_combine.py
>> run pipeline/feature-validation/exp-4/pipeline_validation_patients.py
>> run pipeline/feature-validation/exp-5/pipeline_validation_stacking_wt_mrsi.py
```
