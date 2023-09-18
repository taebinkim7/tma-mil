import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn
import util
import sksurv

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# mil_dir = '/datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he/'
# model_name = 'vgg16'
# layer = 'block4_pool'
# instance_size = 400
# instance_stride = 400
# pool_size = None
# time = 'surv_mos'
# event = 'surv_event'
# group = 'clinical_bins'

parser = argparse.ArgumentParser(description='Extract features.')
parser.add_argument('--mil-dir', required=True, type=str)
parser.add_argument('--model', '-m', required=True, help='CNN model')
parser.add_argument('--layer', '-l', required=True, help='CNN layer')
parser.add_argument('--instance-size', help='instance size')
parser.add_argument('--instance-stride', help='instance stride')
parser.add_argument('--pool-size', '-p', help='mean pooling size')
parser.add_argument('--time', required=True, type=str)
parser.add_argument('--event', required=True, type=str)
parser.add_argument('--group', help='Class groups for reporting results')
args = parser.parse_args()
model_name = args.model
layer = args.layer
instance_size = args.instance_size
instance_stride = args.instance_stride
pool_size = args.pool_size
mil_dir = args.mil_dir
time = args.time
event = args.event
group = args.group

if len(mil_dir) > 1 and mil_dir[-1] != '/':
    mil_dir += '/'

def five_fold_cv(X, y, model_class, n_folds=5):
    kf = sklearn.model_selection.KFold(n_splits=n_folds)
    res = {'c_index': []}
    for train_index, test_index in kf.split(X):
        model = model_class()
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        res['c_index'].append(model.score(X_test, y_test))
    c_index_mean = np.round(np.mean(res['c_index']), 3)
    c_index_sem = np.round(np.std(res['c_index'])/np.sqrt(n_folds), 3)

    print(f"""
        C-index
        {c_index_mean} ({c_index_sem})
        """)
    return

np.random.seed(111)

# load filenames
sample_images = util.load_sample_images(mil_dir)

# read in CNN features
feats = {}
for sample, imagelist in sample_images.items():
    feats[sample] = []
    for fn in imagelist:
        feat_fn = mil_dir + fn[:fn.rfind('.')] + '_' + model_name + '-' + layer
        if pool_size is not None:
            feat_fn += '_p' + str(pool_size)
        if instance_size is not None:
            feat_fn += '_i' + str(instance_size)
        if instance_stride is not None:
            feat_fn += '-' + str(instance_stride)
        feat_fn += '.npy'
        feat = np.load(feat_fn)
        if len(feat) == 0:
            continue
        feats[sample].append(feat)
    feats[sample] = np.concatenate(feats[sample], axis=0)
    if len(feats[sample].shape) == 1:
        feats[sample] = feats[sample].reshape((1, len(feats[sample])))
    # compute mean if needed
    if len(feats[sample].shape) > 1:
            feats[sample] = feats[sample].mean(axis=0)


X = pd.DataFrame(feats).T
label_cols = [event, time]
if group is not None:
    label_cols += [group]

y = pd.read_csv(os.path.join(mil_dir, 'labels.csv'), index_col=0)[label_cols].dropna()
# y['surv'] = [(indicator == 1, time) for indicator, time in zip(y[event], y[time])]
# y = y['surv']

if group is not None:
    ln = np.unique(y[group])
    ln.sort()
    ln = list(ln)
    if '' in ln:
        del ln[ln.index('')]
    label_names_group = ln

inter_idx = list(set(X.index).intersection(set(y.index)))

X = X.loc[inter_idx]
y = y.loc[inter_idx]
# model_class = CoxPHSurvivalAnalysis
model_class = CoxnetSurvivalAnalysis

if group is None:
    y = sksurv.util.Surv.from_dataframe(event, time, y)
    five_fold_cv(X, y, model_class)
else:
    for g in label_names_group:
        X_g = X[y[group] == g]
        y_g = y[y[group] == g]
        y_g = sksurv.util.Surv.from_dataframe(event, time, y_g)
        print(f'Group: {g}')
        print(y_g.shape)
        five_fold_cv(X_g, y_g, model_class)
