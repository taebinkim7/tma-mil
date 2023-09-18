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
parser.add_argument('--train-dir', required=True, help='train directory')
parser.add_argument('--test-dir', required=True, help='test directory')
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
train_dir = args.train_dir
if len(train_dir) > 1 and train_dir[-1] != '/':
    train_dir += '/'
test_dir = args.test_dir
if len(test_dir) > 1 and test_dir[-1] != '/':
    test_dir += '/'
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

# load model from train directory
# TODO: add level
# if level == 'bag':
#     model = load(out_dir + '_' + classifier + '_' + cat)
# elif level == 'instance':
#     model = load(out_dir + '_refitted_' + classifier + '_' + cat)
# model = load(out_dir + '_refitted_' + classifier + '_' + cat)
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


def flatten_bag_data(X, y):
    bags = [np.asmatrix(bag) for bag in X]
    X = np.vstack(bags)
    y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1))) for bag, cls in zip(bags, y)])

    return np.array(X), np.array(y).reshape(-1)

X_test, y_test = flatten_bag_data(X_test, y_test)


def cal_scores(X, level=level):
    coef = model._model.coef_
    intercept = model._model.intercept_

    # calculate classification scores (decision function)
    clf_scores = X @ coef.T + intercept  # probably due to sklearn update?
    clf_scores = clf_scores.reshape(-1)

    # calculate orthogonal PC scores
    X1 = X - X @ coef.T @ coef / (coef @ coef.T)
    pca = PCA(n_components=2).fit(X1)
    opc1 = pca.components_[0]
    opc2 = pca.components_[1]
    opc1_scores = X @ opc1
    opc2_scores = X @ opc2

    return clf_scores, opc1_scores, opc2_scores


# plot scores colored by labels
clf_scores, opc1_scores, opc2_scores = cal_scores(X_test)
pos_idx = (y_test == 1)
neg_idx = (y_test == 0)


# plt.scatter(clf_scores[pos_idx], opc1_scores[pos_idx], c='red', s=3, alpha=.3, label='pos')
# plt.scatter(clf_scores[neg_idx], opc1_scores[neg_idx], c='blue', s=3, alpha=.3, label='neg')
# plt.axvline(x=0, alpha=.5, color='black')
# plt.title('OPC1 score vs. ' + classifier.upper() + ' score')
# plt.xlabel(classifier.upper() + ' score')
# plt.ylabel('OPC1 score')
#
# # save plot
# plt.savefig(out_dir + '_opc1_' + classifier + '_' + cat + '.png')
# plt.close()

scores = [(classifier.upper(), clf_scores), ('OPC1', opc1_scores), ('OPC2', opc2_scores)]
n = len(y_test)
noise = (np.arange(n) - n // 2) / (3 * n)

# generate upper triangular subplots of orthogonal PC scores
fig, axes = plt.subplots(figsize=(10, 10), ncols=3, nrows=3)
for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        if i > j:
            ax.axis('off')
        elif i == j:
            ax.set_xlim([min(scores[i][1]), max(scores[i][1])])
            ax.scatter(scores[i][1][pos_idx], y_test[pos_idx] + noise[pos_idx], c='crimson', s=3, alpha=.3,
                       label='pos' if i == 0 else '')
            ax.scatter(scores[i][1][neg_idx], y_test[neg_idx] + noise[neg_idx], c='royalblue', s=3, alpha=.3,
                       label='neg' if i == 0 else '')
            ax1 = ax.twinx()
            sns.kdeplot(x=scores[i][1][pos_idx], ax=ax1, c='red')
            sns.kdeplot(x=scores[i][1][neg_idx], ax=ax1, c='blue')
            ax1.set_ylabel('')
            ax1.set_yticks([])
            ax2 = ax1.twinx()
            ax2.text(.5, .5, scores[i][0], ha='center', va='center', fontsize=20, transform=ax.transAxes,
                     bbox=dict(boxstyle="square", ec='black', fc='whitesmoke'))
            ax2.set_yticks([])
        elif i < j:
            ax.set_xlim([min(scores[j][1]), max(scores[j][1])])
            ax.scatter(scores[j][1][pos_idx], scores[i][1][pos_idx], c='crimson', s=3, alpha=.3)
            ax.scatter(scores[j][1][neg_idx], scores[i][1][neg_idx], c='royalblue', s=3, alpha=.3)

fig.legend(loc='lower left', fontsize=15)

# save plot
train_group = train_dir.split('/')[-4]
test_group = test_dir.split('/')[-4]

os.makedirs(output_dir, exist_ok=True)
fig.savefig(output_dir + 'cross_opc_plot_' + test_group + '_' + classifier + '_' + cat + '.png')

