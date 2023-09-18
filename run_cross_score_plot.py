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


parser = argparse.ArgumentParser(description='Generate orthogonal PC plot of trained classifier at instance level.')
parser.add_argument('--output-dir', required=True, help='output directory')
parser.add_argument('--train-dirs', nargs='+', required=True, help='train directory')
parser.add_argument('--test-dirs', nargs='+', required=True, help='test directory')
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
train_dirs = args.train_dirs
for i, train_dir in enumerate(train_dirs):
    if len(train_dir) > 1 and train_dir[-1] != '/':
        train_dir += '/'
    train_dirs[i] = train_dir
test_dirs = args.test_dirs
for i, test_dir in enumerate(test_dirs):
    if len(test_dir) > 1 and test_dir[-1] != '/':
        test_dir += '/'
    test_dirs[i] = test_dir
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
        clf_scores = X @ coef.T + intercept
        clf_scores = clf_scores.reshape(-1)

        # calculate orthogonal PC scores
        X1 = X - X @ coef.T @ coef / (coef @ coef.T)
        pca = PCA(n_components=2).fit(X1)
        opc1 = pca.components_[0]
        opc1_scores = X @ opc1

    elif level == 'bag':
        # calculate classification scores at bag level
        clf_scores = model.predict(X)[:, 1]

        qf = model.get_qf(X)
        coef = model._model_agg[0]._model.coef_
        X1 = qf - qf @ coef.T @ coef / (coef @ coef.T)
        pca = PCA(n_components=2).fit(X1)
        opc1 = pca.components_[0]
        opc1_scores = qf @ opc1

    return clf_scores, opc1_scores, y


# generate subplots of scores. First row represents instance level and second row represents bag level
k = len(train_dirs)
titles = ['Original', 'Normalization', 'Augmentation', 'Mix-up', 'SAN']
fig, axes = plt.subplots(figsize=(5 * k, 5 * 2), ncols=k, nrows=2)
for i in range(k):
    train_dir, test_dir = train_dirs[i], test_dirs[i]

    # load model from train directory
    model_path = train_dir + '_' + mi_type + '_' + classifier + '_' + cat + '_i' + str(instance_size) + '-' \
                 + str(instance_stride) + '_q' + str(quantiles)
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

    # # first row: instance level
    # clf_scores, _, instance_y_test = cal_scores(model, X_test, y_test, 'instance')
    # pos_idx = (instance_y_test == 1)
    # neg_idx = (instance_y_test == 0)
    # n = len(instance_y_test)
    # noise = (np.arange(n) - n // 2) / (3 * n)
    #
    # ax = axes[0, i]
    # ax.set_title(titles[i])
    # if i == 0:
    #     ax.set_yticks([0, 1])
    #     ax.set(ylabel='Instance-level')
    # else:
    #     ax.set_yticks([])
    # ax.scatter(clf_scores[pos_idx], instance_y_test[pos_idx] + noise[pos_idx], c='crimson', s=3, alpha=.3,
    #            label='pos' if i == 0 else '')
    # ax.scatter(clf_scores[neg_idx], instance_y_test[neg_idx] + noise[neg_idx], c='royalblue', s=3, alpha=.3,
    #            label='neg' if i == 0 else '')
    # ax1 = ax.twinx()
    # sns.kdeplot(x=clf_scores[pos_idx], ax=ax1, c='red')
    # sns.kdeplot(x=clf_scores[neg_idx], ax=ax1, c='blue')
    # ax1.set_ylabel('')
    # ax1.set_yticks([])

    # compute scores
    clf_scores, opc1_scores, _ = cal_scores(model, X_test, y_test, 'bag')

    # first row: bag-level OPC vs SVM scores
    pos_idx = (y_test == 1)
    neg_idx = (y_test == 0)
    ax = axes[0, i]
    ax.set_title(titles[i])
    ax.set_xlim([0, 1])
    if i == 0:
        ax.set_yticks([])
        ax.set(ylabel='OPC1')
    else:
        ax.set_yticks([])
    ax.scatter(clf_scores[pos_idx], opc1_scores[pos_idx],
               c='tab:orange',
               s=3, alpha=.3,
               label='pos' if i == 0 else '')
    ax.scatter(clf_scores[neg_idx], opc1_scores[neg_idx],
               c='tab:blue',
               s=3, alpha=.3,
               label='neg' if i == 0 else '')

    # second row: bag level SVM scores
    # clf_scores, _, y_test = cal_scores(model, X_test, y_test, 'bag')
    # pos_idx = (y_test == 1)
    # neg_idx = (y_test == 0)
    n = len(y_test)
    noise = (np.arange(n) - n // 2) / (3 * n)
    # pos_p = sum(pos_idx) / (sum(pos_idx) + sum(neg_idx))
    # neg_p = sum(neg_idx) / (sum(pos_idx) + sum(neg_idx))

    ax = axes[1, i]
    if i == 0:
        ax.set_yticks([])
        # ax.set_yticks([0, 1])
        # ax.set(ylabel='Label')
    else:
        ax.set_yticks([])
    ax.scatter(clf_scores[pos_idx], y_test[pos_idx] + noise[pos_idx],
               c='tab:orange',
               s=3, alpha=.3, label='')
    ax.scatter(clf_scores[neg_idx], y_test[neg_idx] + noise[neg_idx],
               c='tab:blue',
               s=3, alpha=.3, label='')

    # ax1 = ax.twinx()
    # sns.kdeplot(x=clf_scores[pos_idx], ax=ax1, c='red')
    # ax1.set_ylim([0, 4/pos_p])
    # ax1.set_ylabel('')
    # ax1.set_yticks([])
    #
    # ax2 = ax.twinx()
    # sns.kdeplot(x=clf_scores[neg_idx], ax=ax2, c='blue')
    # ax2.set_ylim([0, 4/neg_p])
    # ax2.set_ylabel('')
    # ax2.set_yticks([])

    ax1 = ax.twinx()
    sns.kdeplot(x=clf_scores, ax=ax1, hue=y_test)
    ax1.legend([], [], frameon=False)

fig.legend(loc='lower left', fontsize=15)

# save plot
train_group = train_dirs[0].split('/')[-4][-4:]
test_group = test_dirs[0].split('/')[-4][-4:]
os.makedirs(output_dir, exist_ok=True)
fig.tight_layout()
fig.savefig(output_dir + 'cross_score_plot' + '_train_' + train_group + '_test_' + test_group + '_' + classifier + '_' + cat + '.png')
