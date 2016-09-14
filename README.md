Multi-parametric MRI for prostate cancer detection
==================================================

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

- Extract the intensity signal from T2W and ADC,
- Extract the enhancement signal from the DCE,

#### Run the pipeline

To extract the different feature, launch an `ipython` or `python` prompt and run from the root directory:

```python
>> run pipeline/feature-extraction/t2w/pipeline_extraction_intensity_t2w.py
>> run pipeline/feature-extraction/t2w/pipeline_extraction_edge_t2w.py
```
