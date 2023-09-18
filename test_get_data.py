import os
import sys
import argparse
import warnings
import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sksurv.metrics
from joblib import dump, load

import util
from linear_classifier import LinearClassifier
from sil import SIL


out_dir = '/datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he/'
categories = 'er'
model_name = 'vgg16'
layer = 'block4_pool'
instance_size = 400
instance_stride = 400
mi_type = 'quantile'
cv_folds = 5
random_state = 111

# load filenames and labels
sample_images = util.load_sample_images(out_dir)
samples, cats, labels = util.load_labels(out_dir)

# get labels for list of categories
label_names = []
categories = categories.split(',')
new_labels = np.zeros((labels.shape[0], len(categories)), dtype='int')
for i, cat in enumerate(categories):
    c = np.where(cats == cat)[0][0]
    ln = np.unique([l[c] for l in labels])
    ln.sort()
    ln = list(ln)
    if '' in ln:
        del ln[ln.index('')]
    label_names.append(ln)
    new_labels[:, i] = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])

labels = new_labels
cats = categories

# read in CNN features
feats = {}
for sample, imagelist in sample_images.items():
    feats[sample] = []
    for fn in imagelist:
        feat_fn = out_dir + fn[:fn.rfind('.')] + '_' + model_name + '-' + layer
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
    if mi_type is None or mi_type.lower() == 'none':
        if len(feats[sample].shape) > 1:
            feats[sample] = feats[sample].mean(axis=0)
    # build train/test sets
    idx = np.arange(len(samples))

for c, cat_name in enumerate(cats):
    idx = [i for i in idx if (labels[i, c] != -1) & (samples[i] in feats)]
    labels_subset = labels[idx]
    if len(label_names) == 1:
        skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                                          random_state=random_state)
        idx_train_test = list(skf.split(idx, labels_subset[:, c]))
    else:
        # merge label categories to do stratified folds
        skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        la_all = np.array(labels_subset[:, c])
        p = 1
        for i in range(labels_subset.shape[1]):
            la_all += labels_subset[:, i] * p
            p *= len(label_names[i])
        idx_train_test = list(skf.split(idx, la_all))