#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def cross_val(model, _X, _y, n_repeats, n_splits):
    import numpy as np
    from sklearn import metrics
    from numpy import interp
    from sklearn.model_selection import cross_validate
    import pprint 
    from sklearn.model_selection import RepeatedStratifiedKFold
    n_repeats = n_repeats
    n_splits = n_splits
	
    _cv = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42)
    _scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'neg_brier_score']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)     
    return {"Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Mean Validation Precision": results['test_precision'].mean(),
            "Mean Validation Recall": results['test_recall'].mean(),
            "Mean Validation F1 Score": results['test_f1'].mean(),
            "Mean ROC": results['test_roc_auc'].mean(),
            "Brier": results['test_neg_brier_score'].mean(),
            "Balanced accuracy": results['test_balanced_accuracy'].mean()}  

