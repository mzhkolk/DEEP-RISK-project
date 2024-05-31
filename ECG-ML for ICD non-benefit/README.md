# Optimizing-Patient-Selection-for-Primary-Prevention-ICD-Implantation

## Project aim
This project aims to extract features from ECG time-series data and develop machine learning models to predict the risk of ICD non-benefit.

## Features extraction
The pre-processed ECGs were characterized by 65 unique features calculated using the tsfresh algorithm. The feature selection approach combined the Benjamini-Yekutieli procedure and a recursive feature elimination algorithm.

## Machine learning models
The machine learning models used in this project are:
```
- Support vector machines (SVM)
- Extreme gradient boosting algorithms (XGBoost)
- Random forest (RF) classifiers
```

## Interpretation of predictions
The interpretation of individual predictions was explained using the SHAP method and mean waveforms for predicted high risk and predicted low risk of ICD non-benefit in the external patient cohort were displayed.

## Requirements
```
- Python 3.6
- tsfresh version 0.12.0
- scikit-learn library version 1.1.1
- XGBoost library version 1.6.2
- numpy
- pandas
- h5py 3.7.0
- pydicom
```

## Usage
The detailed description of ECG pre-processing and feature extraction is provided in the Supplemental Methods. To use this project:
```
1. Extract the ECG features using tsfresh version 0.12.0 in Python 3.6
2. Perform the feature selection and model training using the scikit-learn and XGBoost libraries
3. The internal model evaluation was performed by repeated stratified k-fold cross-validation with k=10 and 5 repeats
4. The performance of the final model was evaluated on the external testing cohort
5. The interpretation of individual predictions can be explained using the SHAP method.
```
