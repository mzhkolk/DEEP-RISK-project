#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Extract relevant features
def extract_relevant_features(input_data, outcome, numberoffeatures):
    from tsfresh import extract_features, select_features
    from tsfresh.feature_selection.relevance import calculate_relevance_table
    relevance_table = calculate_relevance_table(input_data, outcome, fdr_level=0.05, n_significant=4)
    top = relevance_table.iloc[:numberoffeatures,:]
    input = top['feature']
    list_of_relevantfeatures = (list(input))
    features_filtered = input_data[list_of_relevantfeatures]
    return features_filtered

def drop_input_corr_columns(x_features, corr_fac = 0.90):
    import numpy as np
    # Create correlation matrix
    corr_matrix = x_features.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than corr_fac
    to_drop = [column for column in upper.columns if any(upper[column] > corr_fac)]
    # Drop features 
    x_features_no_colnr = x_features.drop(columns = to_drop) 
    return x_features_no_colnr, to_drop


#L1 LogisticRegression
def lassoL1(input_data, outcome, numberoffeatures):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(penalty='l1', random_state = 1201, solver='liblinear',
                             multi_class = 'ovr', C = 1, max_iter = 1000).fit(input_data, outcome)
    logReg_coef = log_reg.coef_
    topFeaturesIdx = np.argsort(np.abs(logReg_coef[0,:]))[-numberoffeatures:]
    #print("The most relevant features:", list(input_data.columns[topFeaturesIdx]))
    ecg_rem =input_data.columns[topFeaturesIdx]
    X_internal_ecg = input_data.iloc[:,topFeaturesIdx]
    return X_internal_ecg