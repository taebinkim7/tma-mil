import os
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='Generate orthogonal PC plot of trained classifier at instance level.')
parser.add_argument('--out_dir', '-o', required=True, help='output directory')
parser.add_argument('--cat', help='label category used to train')
parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, or dwd); default: svm')
parser.add_argument('--level', '-l', help='level of classification (bag, instance)')
parser.add_argument('--random-state', help='random state for splitting datasets')
args = parser.parse_args()
out_dir = args.out_dir
if len(out_dir) > 1 and out_dir[-1] != '/':
    out_dir += '/'
cat = args.cat
classifier = args.classifier
random_state = args.random_state
level = args.level

data = load(out_dir + '_data_' + cat + '_random_state_' + random_state)

# TODO: add level
# if level == 'bag':
#     model = load(out_dir + '_' + classifier + '_' + cat)
# elif level == 'instance':
#     model = load(out_dir + '_refitted_' + classifier + '_' + cat)
model = load(out_dir + '_refitted_' + classifier + '_' + cat)

(X_train, y_train), (X_test, y_test) = data


def flatten_bag_data(X, y):
    bags = [np.asmatrix(bag) for bag in X]
    X = np.vstack(bags)
    y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1))) for bag, cls in zip(bags, y)])

    return np.array(X), np.array(y).reshape(-1)


X_train, y_train = flatten_bag_data(X_train, y_train)
X_test, y_test = flatten_bag_data(X_test, y_test)


def cal_scores(X, level=level):
    # calculate classification scores (decision function)
    clf_scores = X @ model.coef_.T + model.intercept_
    clf_scores = clf_scores.reshape(-1)

    # calculate orthogonal PC scores
    X1 = X - X @ model.coef_.T @ model.coef_ / (model.coef_ @ model.coef_.T)
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
fig.savefig(out_dir + 'plots/_opc_plot_' + classifier + '_' + cat + '.png')

