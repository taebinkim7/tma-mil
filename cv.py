import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def five_fold_cv(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits)
    res = {'acc': [], 'sensitivity': [], 'specificity': [], 'auc': []}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        res['acc'].append(model.score(X_test, y_test))
        y_pred = model.predict_proba(X_test)[:, 1]
        res['sensitivity'].append(float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
        res['specificity'].append(float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
        res['auc'].append(roc_auc_score(y_test, y_pred))
    print(f"""
        Accuracy Sensitivity Specificity AUC
        {np.mean(res['acc'])},{np.mean(res['sensitivity'])},{np.mean(res['specificity'])},{np.mean(res['auc'])}
        {np.std(res['acc'])/np.sqrt(n_splits)},{np.std(res['sensitivity'])/np.sqrt(n_splits)},{np.std(res['specificity'])/np.sqrt(n_splits)},{np.std(res['auc'])/np.sqrt(n_splits)}
        """)
    return

