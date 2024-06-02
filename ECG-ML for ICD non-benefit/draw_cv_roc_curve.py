#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def draw_cv_roc_curve(classifier, cv, X, y):
    import numpy as np
    from numpy import interp
    classifier = classifier
    cv = cv
    X = X
    y = y
    from sklearn import metrics
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import RocCurveDisplay
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])              
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.8,
        #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=0.5, color='grey',
             label=None)    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='navy', label=r'Internal validation: Mean ROC (AUROC=%0.3f $\pm$%0.2f)' % (mean_auc,std_auc), lw=2, alpha=.6)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1) #, label=r'$\pm$1 SD')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
   

