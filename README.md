# NEAR_python

Artifact removal pipeline for human newborn EEG data and mobile EEG for BCI applications

Python implementation of the Matlab public repository of NEAR: https://github.com/vpKumaravel/NEAR

This pipeline integrates MNE scripts with additional NEAR artifact rejection steps: pre-liminary calibration of the bad channel detection threshold with Local Outlier Factor (LOF) and of the ASR cut-off parameter, bad channel detection using LOF algorithm and correction/removal of bad segments using ASR. In addition, we provided the scripts (based on MNE functions) for a fully automated EEG processing from raw to clean data: importing and filtering raw data, interpolation of removed channels and re-referencing. 

* To familiarize with the user parameters, execute the step-by-step preprocessing using "NEAR_tutorial.py". 

* To run the full NEAR pipline for a single subject EEG file, please use "NEAR_main.py". 
