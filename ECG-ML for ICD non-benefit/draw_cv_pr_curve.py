#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def draw_cv_pr_curve(classifier, cv, X, y):
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
    prs = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        pr, re, thresholds  = precision_recall_curve(y.iloc[test], probas_[:, 1])     
        prs.append(interp(mean_recall, pr, re))
        #prs[-1][0] = 0.0
        pr_auc = metrics.auc(re, pr)
        aucs.append(pr_auc)    
        # Plotting each individual PR Curve
       # plt.plot(re, pr, lw=3, alpha=0.5, label='Fold %d (AUCPR = %0.2f)' % (i+1, pr_auc))
        i += 1
   # plt.plot([0, 1], [1, 0], linestyle='--', lw=3, color='k', label=None, alpha=.6)
    mean_precision = np.mean(prs, axis=0)
    #mean_precision[-1] = 1.0
    mean_auc = metrics.auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    plt.plot(mean_recall, mean_precision, color='navy', label=r'Internal validation: Mean PRC (AUPRC=%0.3f $\pm$%0.2f)' % (mean_auc, std_auc), lw=2,alpha=.6)
    std_prs = np.std(prs, axis=0)
    tprs_upper = np.minimum(mean_precision + std_prs, 1)
    tprs_lower = np.maximum(mean_precision - std_prs, 0)
    plt.fill_between(mean_recall, tprs_lower, tprs_upper, color='grey', alpha=.1) #, label=r'$\pm$ 1 SD')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    

