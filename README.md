![Pep8](https://github.com/ramp-kits/brain_age_with_site_removal/actions/workflows/pep8.yml/badge.svg)
![Testing Conda](https://github.com/ramp-kits/brain_age_with_site_removal/actions/workflows/testing_conda.yml/badge.svg)
![Testing Pip](https://github.com/ramp-kits/brain_age_with_site_removal/actions/workflows/testing_pip.yml/badge.svg)
![Testing Notebook](https://github.com/ramp-kits/brain_age_with_site_removal/actions/workflows/testing_notebook.yml/badge.svg)

![Challenge](https://baobablab.github.io/bhb/images/resources/age_prediction_with_site_removal.jpg)

# Brain age prediction with site-effect removal

The [challenge](https://baobablab.github.io/bhb/challenges/age_prediction_with_site_removal) uses the openBHB dataset and aims to i) predict age from derived data from 3D T1 anatomical MRI while ii) removing site information from the learned representation. Thus, we aim to compare the capacity of proposed models to encode a relevant representation of the data (feature extraction and dimensionality reduction) that preserve the biological variability associated with age while removing the site-specific information. The algorithms submitted must output a low-dimension features vector (p < 10000). Derived data are composed of Quasi-Raw, VBM, and SBM.

## A big data challenge

[OpenBHB](https://baobablab.github.io/bhb/dataset#healthy-controls-datasets) aggregates 10 publicly available datasets. Currently, openBHB is focused only on Healthy Controls (HC) since the main challenge consists in modeling the (normal) brain development by building a robust brain age predictor. OpenBHB contains $N=5330$ brain MRI scans from HC acquired on 71 different acquisition sites coming from European-American, European, and Asian individuals, promoting more diversity in openBHB. To manage redundant images, one session per participant has been retained along with its best-associated run, selected according to image quality. We also provide the participants phenotype as well as site and scanner information associated with each image, which essentially includes age, sex, acquisition site, diagnosis (in our case only HC), MRI scanner magnetic field, and MRI scanner settings identifier (a combination of multiple information composed of a subset of the repetition time, echo time, sequence name, flip angle, and acquisition coil). Some widespread confounds are also proposed, such as the Total Intracranial Volume (TIV), the CerebroSpinal Fluid Volume (CSFV), the Gray Matter Volume (GMV), and the White Matter Volume (WMV).

![Population Statistics](https://baobablab.github.io/bhb/images/resources/population.png)

## Multi-modal imaging data

For the moment only features derived from T1w images are available comprising Quasi-Raw, CAT12 VBM, and FreeSurfer. All data are [preprocessed uniformly](https://baobablab.github.io/bhb/dataset#pre-processed-datasets) including a semi-automatic Quality Controls (QC) guided with quality metrics.

![BrainPrep preprocessings](https://baobablab.github.io/bhb/images/resources/preproc.png)

## Coding framework, for competition and collaboration

The challenge will be carried out on the [RAMP platform](https://ramp.studio). It enables competition and collaboration on data-science problems, using the Python language. To start "hacking", a [starting kit](https://github.com/ramp-kits/brain_age_with_site_removal/brain_age_with_site_removal_starting_kit.ipynb) is available. It provides a simple working example which can be expanded to more advanced solutions.

## For developers

The behaviour of the code can be controlled from three environment variables:

- RAMP_BRAIN_AGE_SITERM_TEST: set this environment varaible to 'on' in order to work on the test dataset. Under the hood, it will select the appropriate target labels and use a KFold cross-validation.
- RAMP_BRAIN_AGE_SITERM_SMALL: set this environment varaible to 'on' in order to select a small part of the dataset. This option is usefull when testing the challenge on the server side. 
- RAMP_BRAIN_AGE_SITERM_CACHE:  set this environment varaible to 'on' in order to save some intermediate results in the problem folder that in turn are used as a hard-caching system.
- RAMP_BRAIN_AGE_SITERM_SERVER: set this environment varaible to 'on' in order to select the proper options when deploying the event on the server side (i.e. the train datasets don't have the same number of labels).
