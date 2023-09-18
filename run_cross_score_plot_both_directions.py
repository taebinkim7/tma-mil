import os
import sys
import warnings
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA

import util


parser = argparse.ArgumentParser(description='Generate orthogonal PC plot of trained classifier.')
parser.add_argument('--output-dir', required=True, help='output directory')
parser.add_argument('--train-dirs1', nargs='+', required=True, help='train directory')
parser.add_argument('--test-dirs1', nargs='+', required=True, help='test directory')
parser.add_argument('--train-dirs2', nargs='+', required=True, help='train directory')
parser.add_argument('--test-dirs2', nargs='+', required=True, help='test directory')
parser.add_argument('--instance-size', help='train instance size')
parser.add_argument('--instance-stride', help='train instance stride')
parser.add_argument('--cat', help='label category used to train')
parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, or dwd); default: svm')
parser.add_argument('--mi', help='MI type (none, median, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
parser.add_argument('--quantiles', '-q', help='Number of quantiles')
parser.add_argument('--level', '-l', help='level of classification (bag, instance)')
args = parser.parse_args()
output_dir = args.output_dir
if len(output_dir) > 1 and output_dir[-1] != '/':
    output_dir += '/'
train_dirs1 = args.train_dirs1
for i, train_dir in enumerate(train_dirs1):
    if len(train_dir) > 1 and train_dir[-1] != '/':
        train_dir += '/'
    train_dirs1[i] = train_dir
test_dirs1 = args.test_dirs1
for i, test_dir in enumerate(test_dirs1):
    if len(test_dir) > 1 and test_dir[-1] != '/':
        test_dir += '/'
    test_dirs1[i] = test_dir
train_dirs2 = args.train_dirs2
for i, train_dir in enumerate(train_dirs2):
    if len(train_dir) > 1 and train_dir[-1] != '/':
        train_dir += '/'
    train_dirs2[i] = train_dir
test_dirs2 = args.test_dirs2
for i, test_dir in enumerate(test_dirs2):
    if len(test_dir) > 1 and test_dir[-1] != '/':
        test_dir += '/'
    test_dirs2[i] = test_dir
instance_size = args.instance_size
instance_stride = args.instance_stride
cat = args.cat
classifier = args.classifier
mi_type = args.mi
quantiles = args.quantiles
level = args.level

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def flatten_bag_data(X, y):
    bags = [np.asmatrix(bag) for bag in X]
    X = np.vstack(bags)
    y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1))) for bag, cls in zip(bags, y)])

    return np.array(X), np.array(y).reshape(-1)


def cal_scores(model, X, y, level=level):
    if level == 'instance':
        # 'SIL' object has no attribute 'coef_'. Probably due to sklearn update?
        coef = model._model.coef_
        intercept = model._model.intercept_

        # calculate classification scores at instance level
        X, y = flatten_bag_data(X, y)
        clf_scores = X @ coef.T - intercept
        clf_scores = clf_scores.reshape(-1)

    elif level == 'bag':
        # calculate classification scores at bag level
        # clf_scores = model.predict(X)[:, 1]

        qf = model.get_qf(X)
        coef = model._model_agg[0]._model.coef_
        intercept = model._model_agg[0]._model.intercept_

        clf_scores = qf @ coef.T
        clf_scores = clf_scores.reshape(-1)

        clf_scores = (clf_scores - np.mean(clf_scores, axis=0)) / np.std(clf_scores, axis=0)

    return clf_scores, y

def get_clf_scores(train_dir, test_dir):
    # load model from train directory
    model_path = train_dir + '_' + mi_type + '_' + classifier + '_' + cat + '_i' + str(instance_size) + '-' \
                 + str(instance_stride) + '_q' + str(quantiles) \
                 + '_fold3'
    model = load(model_path)

    # load data from test directory
    test_sample_images = util.load_sample_images(test_dir)
    test_samples, test_cats, test_labels = util.load_labels(test_dir)
    test_cats, test_labels, test_label_names = util.clean_cats_labels(test_cats, test_labels, cat)
    test_feats = util.load_feats(test_dir, test_sample_images, 'vgg16', 'block4_pool', None, instance_size, instance_stride,
                                 mi_type)
    idx_test = np.arange(len(test_samples))
    idx_test = [i for i in idx_test if (test_labels[i, 0] != -1) & (test_samples[i] in test_feats)]
    X_test = [test_feats[test_samples[i]] for i in idx_test]
    y_test = test_labels[idx_test, 0]

    # compute scores
    clf_scores, _ = cal_scores(model, X_test, y_test, 'bag')

    return clf_scores, y_test


# generate subplots of scores. First row represents train to test and second represents test to train.
k = len(train_dirs1)
titles = ['Original', 'Normalization', 'Augmentation', 'Mix-up', 'SAN']
fig, axes = plt.subplots(figsize=(5 * k, 5 * 2), ncols=k, nrows=2)
for i in range(k):
    # first row: bag level SVM scores train to test
    # scores = [0.779, 0.809, 0.816, 0.820, 0.854]

    train_dir, test_dir = train_dirs1[i], test_dirs1[i]
    clf_scores, y_test = get_clf_scores(train_dir, test_dir)

    pos_idx = (y_test == 1)
    neg_idx = (y_test == 0)

    n = len(y_test)
    noise = (np.arange(n) - n // 2) / (3 * n)

    ax = axes[0, i]
    ax.set_title(titles[i], fontsize=20)
    ax.set_xlim([-3, 3])
    ax.set_yticks([])

    ax.scatter(clf_scores[pos_idx], y_test[pos_idx] + noise[pos_idx],
               c='tab:orange',
               s=5, alpha=.5,
               label='pos' if i == 0 else '')
    ax.scatter(clf_scores[neg_idx], y_test[neg_idx] + noise[neg_idx],
               c='tab:blue',
               s=5, alpha=.5,
               label='neg' if i == 0 else '')

    # generate KDE plot
    ax1 = ax.twinx()
    sns.kdeplot(x=clf_scores, ax=ax1, hue=y_test)
    ax1.legend([], [], frameon=False)
    # ax2 = ax1.twinx()
    # ax2.text(.5, .5, 'AUC: ' + str(scores[i]), ha='center', va='center', fontsize=20, transform=ax.transAxes,
    #          bbox=dict(boxstyle="square", ec='black', fc='whitesmoke'))
    # ax2.set_yticks([])

    # second row: bag level SVM scores test to train
    # scores = [0.695, 0.706, 0.744, 0.746, 0.775]

    train_dir, test_dir = train_dirs2[i], test_dirs2[i]
    clf_scores, y_test = get_clf_scores(train_dir, test_dir)

    pos_idx = (y_test == 1)
    neg_idx = (y_test == 0)

    n = len(y_test)
    noise = (np.arange(n) - n // 2) / (3 * n)

    ax = axes[1, i]
    ax.set_xlim([-3, 3])
    ax.set_yticks([])

    ax.scatter(clf_scores[pos_idx], y_test[pos_idx] + noise[pos_idx],
               c='tab:orange',
               s=5, alpha=.5, label='')
    ax.scatter(clf_scores[neg_idx], y_test[neg_idx] + noise[neg_idx],
               c='tab:blue',
               s=5, alpha=.5, label='')

    # generate KDE plot
    ax1 = ax.twinx()
    sns.kdeplot(x=clf_scores, ax=ax1, hue=y_test)
    ax1.legend([], [], frameon=False)
    # ax2 = ax1.twinx()
    # ax2.text(.5, .5, 'AUC: ' + str(scores[i]), ha='center', va='center', fontsize=20, transform=ax.transAxes,
    #          bbox=dict(boxstyle="square", ec='black', fc='whitesmoke'))
    # ax2.set_yticks([])

fig.legend(loc='upper left', fontsize=15)

# save plot
train_group = train_dirs1[0].split('/')[-4][-4:]
test_group = test_dirs1[0].split('/')[-4][-4:]
os.makedirs(output_dir, exist_ok=True)
fig.tight_layout()
fig.savefig(output_dir + 'cross_score_plot' + '_' + train_group + '_' + test_group + '_' + classifier + '_' + cat + '.png')
