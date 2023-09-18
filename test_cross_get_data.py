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


train_dir = '/datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he/'
test_dir = '/datastore/lbcfs/labs/smarronlab/tkim/data/tma_9741/mil/he/'
categories = 'er'
model_name = 'vgg16'
layer = 'block4_pool'
train_instance_size = test_instance_size = 400
train_instance_stride = test_instance_stride = 400
mi_type = 'quantile'
pool_size = None
cv_folds = 5
random_state = 111


# load filenames and labels
train_sample_images = util.load_sample_images(train_dir)
train_samples, train_cats, train_labels = util.load_labels(train_dir)
test_sample_images = util.load_sample_images(test_dir)
test_samples, test_cats, test_labels = util.load_labels(test_dir)

# clean labels
train_cats, train_labels, train_label_names = util.clean_cats_labels(train_cats, train_labels, categories)
test_cats, test_labels, test_label_names = util.clean_cats_labels(test_cats, test_labels, categories)
assert train_cats == test_cats, 'Categories not matched.'
cats = test_cats
label_names = test_label_names

# read in CNN features
train_feats = util.load_feats(train_dir, train_sample_images, model_name, layer, pool_size, train_instance_size,
                              train_instance_stride, mi_type)
test_feats = util.load_feats(test_dir, test_sample_images, model_name, layer, pool_size, test_instance_size,
                             test_instance_stride, mi_type)

# train classifier
idx_train = np.arange(len(train_samples))
idx_test = np.arange(len(test_samples))
for c, cat_name in enumerate(cats):
    print(cat_name)
    idx_train = [i for i in idx_train if (train_labels[i, c] != -1) & (train_samples[i] in train_feats)]
    idx_test = [i for i in idx_test if (test_labels[i, c] != -1) & (test_samples[i] in test_feats)]
    train_labels_subset = train_labels[idx_train]
    test_labels_subset = test_labels[idx_test]
    skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    idx_train_list = list(skf.split(idx_train, train_labels_subset[:, c]))
    idx_test_list = list(skf.split(idx_test, test_labels_subset[:, c]))
    check_idx = []
    check_test_idx = set()
    for f in range(cv_folds):
        idx_train_fold, _ = idx_train_list[f]
        _, idx_test_fold = idx_test_list[f]
        idx_train_fold = np.array(idx_train)[idx_train_fold]
        idx_test_fold = np.array(idx_test)[idx_test_fold]
        check_idx.append(set(np.concatenate((idx_train_fold, idx_test_fold))))
        check_test_idx.update(idx_test_fold)
