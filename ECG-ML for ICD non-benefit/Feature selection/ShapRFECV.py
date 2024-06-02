#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def ShapRFECV(X, y, steps):
    X = X
    y = y
    steps = steps
    X = X.to_numpy()
    y = y.to_numpy()
    from probatus.feature_elimination import ShapRFECV
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    import numpy as np
    import pandas as pd
    import lightgbm
    clf = lightgbm.LGBMClassifier(max_depth=5, class_weight='balanced')
    param_grid = {
        'n_estimators': [5, 7, 10],
        'num_leaves': [3, 5, 7, 10],
    }
    search = RandomizedSearchCV(clf, param_grid)
    search = RandomizedSearchCV(clf, param_grid, cv=5, scoring='roc_auc', refit=False, random_state=42)
    shap_elimination = ShapRFECV(search, step=steps, cv=10, scoring='roc_auc', n_jobs=3, random_state=42)
    report = shap_elimination.fit_compute(X, y, check_additivity=False)
    #print(report)
    return shap_elimination, shap_elimination.plot(), report



