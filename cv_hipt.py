import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn

from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


parser = argparse.ArgumentParser(description='Extract features.')
parser.add_argument('--mil-dir', required=True, type=str)
parser.add_argument('--cat', required=True, type=str)
args = parser.parse_args()
mil_dir = args.mil_dir
cat = args.cat

np.random.seed(111)

X = pd.read_csv(os.path.join(mil_dir, 'hipt_features.csv'), index_col=0).dropna()
y = pd.read_csv(os.path.join(mil_dir, 'labels.csv'), index_col=0)[cat].dropna()

inter_idx = list(set(X.index).intersection(set(y.index)))

X = X.loc[inter_idx]
y = y.loc[inter_idx]

def five_fold_cv(X, y, model_class, n_folds=5):
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    res = {'acc': [], 'sensitivity': [], 'specificity': [], 'auc': []}
    for train_index, test_index in kf.split(X):
        Cvals = [float(2 ** e) for e in range(-15, 10)]
        cv = 20
        metric = 'accuracy'
        model = sklearn.model_selection.GridSearchCV(model_class(class_weight='balanced'),
                                                   [{'C': Cvals}], cv=cv, scoring=metric, n_jobs=-1, refit=True)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        res['acc'].append(model.score(X_test, y_test))
        d = model.decision_function(X_test)
        p = 1.0 / (np.exp(-d) + 1)
        if len(p.shape) == 1:
            p = p.reshape((len(p), 1))
            p = np.concatenate((1 - p, p), axis=1)
        p_predict = p
        y_predict = np.argmax(p_predict, axis=1)
        res['sensitivity'].append(float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
        res['specificity'].append(float(np.logical_and(y_test != 1, y_predict == y_test).sum()) / (y_test != 1).sum())
        res['auc'].append(roc_auc_score(y_test, p_predict[:, 1]))
    acc_mean = np.round(np.mean(res['acc']), 3)
    acc_sem = np.round(np.std(res['acc'])/np.sqrt(n_folds), 3)
    sens_mean = np.round(np.mean(res['sensitivity']), 3)
    sens_sem = np.round(np.std(res['sensitivity'])/np.sqrt(n_folds), 3)
    spec_mean = np.round(np.mean(res['specificity']), 3)
    spec_sem = np.round(np.std(res['specificity'])/np.sqrt(n_folds), 3)
    auc_mean = np.round(np.mean(res['auc']), 3)
    auc_sem = np.round(np.std(res['auc'])/np.sqrt(n_folds), 3)

    print(f"""
        Accuracy Sensitivity Specificity AUC
        {acc_mean} ({acc_sem}),{sens_mean} ({sens_sem}),{spec_mean} ({spec_sem}),{auc_mean} ({auc_sem})
        """)
    return

model_class = LinearSVC
five_fold_cv(X, y, model_class)
