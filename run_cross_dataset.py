import os
import sys
import argparse
import warnings
import numpy as np
import sklearn.metrics
from joblib import dump, load

import util
from linear_classifier import LinearClassifier
from sil import SIL

import warnings
warnings.filterwarnings("ignore")


class ResultsReport:
    def __init__(self, label_names=None):
        self.res = {}
        self.label_names = label_names

    def add(self, metric, result):
        if metric not in self.res:
            self.res[metric] = []
        self.res[metric].append(result)

    def print_summary(self, metric=None):
        if metric is None:
            n_folds = len(self.res['acc'])
            acc_mean = np.mean(self.res['acc'])
            acc_sem = np.std(self.res['acc']) / np.sqrt(n_folds)
            sens_mean = np.mean(self.res['sensitivity'])
            sens_sem = np.std(self.res['sensitivity']) / np.sqrt(n_folds)
            spec_mean = np.mean(self.res['specificity'])
            spec_sem = np.std(self.res['specificity']) / np.sqrt(n_folds)
            auc_mean = np.mean(self.res['auc'])
            auc_sem = np.std(self.res['auc']) / np.sqrt(n_folds)

            print("""
                Accuracy Sensitivity Specificity AUC
                {:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f}) & {:.3f} ({:.3f})
                """.format(acc_mean, acc_sem, sens_mean, sens_sem, spec_mean, spec_sem, auc_mean, auc_sem))
            self.print_summary('confusion')
            return
        if metric != 'confusion':
            mean = np.mean(self.res[metric])
            std = np.std(self.res[metric])
            ste = std / np.sqrt(len(self.res[metric]))
            print('%s %f %f %f' % (metric, mean, std, ste))
        else:
            print('confusion')
            print(('%s ' * len(self.label_names)) % tuple(self.label_names))
            print(sum(self.res['confusion']))
            print('Negative / Positive')
            print(np.sum(sum(self.res['confusion']), axis=1))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test MI classifier.')
    parser.add_argument('--train-dir', nargs='+', required=True, help='training data directories')
    parser.add_argument('--test-dir', nargs='+', required=True, help='test data directories')
    parser.add_argument('--model-dir', help='model directory')
    parser.add_argument('--model', '-m', required=True, help='CNN model')
    parser.add_argument('--layer', '-l', required=True, help='CNN layer')
    parser.add_argument('--train-instance-size', help='train instance size')
    parser.add_argument('--train-instance-stride', help='train instance stride')
    parser.add_argument('--test-instance-size', help='test instance size')
    parser.add_argument('--test-instance-stride', help='test instance stride')
    parser.add_argument('--pool-size', '-p', help='mean pooling size')
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all')
    parser.add_argument('--calibrate', action='store_true', help='calibrate classifier')
    parser.add_argument('--metric', help='metric to optimize during parameter search (accuracy, balanced_accuracy, roc_auc); default: accuracy')
    parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, dwd); default: svm')
    parser.add_argument('--kernel', help='SVM kernel; default: linear')
    parser.add_argument('--mi', help='MI type (none, median, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
    parser.add_argument('--agg', help='Aggregate type (svm, dwd); default: svm')
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16')
    parser.add_argument('--cv-fold-files', help='cross-validation fold files')
    parser.add_argument('--cv-folds', help='cross-validation folds')
    parser.add_argument('--cv-lno', help='cross-validation leave n out')
    parser.add_argument('--random-state', help='random state for splitting datasets')
    parser.add_argument('--save-train', action='store_true', help='save trained classifier')
    parser.add_argument('--load-train', action='store_true', help='load trained classifier')
    parser.add_argument('--n-jobs', help='number of parallel threads')
    parser.add_argument('--n-components', help='number of principal components')
    args = parser.parse_args()
    train_dir = args.train_dir
    for i, d in enumerate(train_dir):
        if len(d) > 1 and d[-1] != '/':
            d += '/'
        train_dir[i] = d
    test_dir = args.test_dir
    for i, d in enumerate(test_dir):
        if len(d) > 1 and d[-1] != '/':
            d += '/'
        test_dir[i] = d
    model_dir = args.model_dir
    if model_dir is not None and len(model_dir) > 1 and model_dir[-1] != '/':
        model_dir += '/'
    model_name = args.model
    layer = args.layer
    train_instance_size = args.train_instance_size
    train_instance_stride = args.train_instance_stride
    test_instance_size = args.test_instance_size
    test_instance_stride = args.test_instance_stride
    pool_size = args.pool_size
    categories = args.cat
    metric = args.metric
    calibrate = args.calibrate
    classifier = args.classifier
    kernel = args.kernel
    mi_type = args.mi
    agg_type = args.agg
    quantiles = args.quantiles
    cv_fold_files = args.cv_fold_files
    cv_folds = args.cv_folds
    cv_lno = args.cv_lno
    random_state = args.random_state
    save_train = args.save_train
    load_train = args.load_train
    n_jobs = args.n_jobs
    n_components = args.n_components

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    if calibrate is None:
        calibrate = False
    else:
        calibrate = bool(calibrate)
        print(f'Calibrate: {calibrate}')

    if classifier is None:
        classifier = 'svm'

    if agg_type is None:
        agg_type = 'svm'

    if quantiles is None:
        quantiles = 16

    if cv_folds is not None:
        cv_folds = int(cv_folds)

    if save_train is None:
        save_train = False
    else:
        save_train = bool(save_train)

    if load_train is None:
        load_train = False
    else:
        load_train = bool(load_train)

    if n_jobs is not None:
        n_jobs = int(n_jobs)

    if random_state is None:
        random_state = 111
    else:
        random_state = int(random_state)

    if n_components is not None:
        n_components = int(n_components)

    # load data
    train_feats, train_samples, train_cats, train_labels, train_label_names = util.load_multi_data(train_dir,
        model_name, layer, pool_size, train_instance_size, train_instance_stride, mi_type, categories)
    test_feats, test_samples, test_cats, test_labels, test_label_names = util.load_multi_data(test_dir,
        model_name, layer, pool_size, test_instance_size, test_instance_stride, mi_type, categories)
    assert train_cats == test_cats, 'Categories not matched.'
    cats = test_cats
    label_names = test_label_names

    # set parameters
    options = {}
    if kernel is not None:
        options['kernel'] = kernel
    else:
        options['kernel'] = 'linear'
    if classifier is not None:
        options['classifier'] = classifier
    if mi_type is not None:
        options['predict_type'] = mi_type
        if agg_type is not None:
            options['agg_type'] = agg_type
    if metric is not None:
        options['metric'] = metric
    if n_components is not None:
        options['n_components'] = n_components

    # train classifier
    idx_train = np.arange(len(train_samples))
    idx_test = np.arange(len(test_samples))
    for c, cat_name in enumerate(cats):
        print(cat_name)
        res = ResultsReport(label_names[c])
        idx_train = [i for i in idx_train if (train_labels[i, c] != -1) & (train_samples[i] in train_feats)]
        idx_test = [i for i in idx_test if (test_labels[i, c] != -1) & (test_samples[i] in test_feats)]

        train_labels_subset = train_labels[idx_train]
        test_labels_subset = test_labels[idx_test]

        skf = sklearn.model_selection.StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        idx_train_list = list(skf.split(idx_train, train_labels_subset[:, c]))
        idx_test_list = list(skf.split(idx_test, test_labels_subset[:, c]))
        for f in range(cv_folds):
            idx_train_fold, _ = idx_train_list[f]
            _, idx_test_fold = idx_test_list[f]
            idx_train_fold = np.array(idx_train)[idx_train_fold]
            idx_test_fold = np.array(idx_test)[idx_test_fold]
            X_train = [train_feats[train_samples[i]] for i in idx_train_fold]
            y_train = train_labels[idx_train_fold, c]
            X_test = [test_feats[test_samples[i]] for i in idx_test_fold]
            y_test = test_labels[idx_test_fold, c]

            # load trained classifier
            if len(train_dir) == 1:
                model_dir = train_dir[0]
            model_path = model_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' \
                + str(train_instance_size) + '-' + str(train_instance_stride) + '_q' + str(quantiles) \
                + '_fold' + str(f)
            if calibrate:
                model_path += '_calibrated'
            if load_train and os.path.exists(model_path):
                model = load(model_path)
            elif mi_type is None:
                model = LinearClassifier(n_jobs=n_jobs, **options)
                model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
            elif mi_type in ['median', 'max']:
                model = SIL(n_jobs=n_jobs, **options)
                model.fit(X_train, y_train, calibrate=calibrate, param_search=True)
            elif 'quantile' in mi_type:
                if quantiles is not None:
                    options['quantiles'] = int(quantiles)
                model = SIL(n_jobs=n_jobs, **options)
                model.fit(X_train, y_train, calibrate=calibrate, param_search=True)

            p_predict = model.predict(X_test)
            y_predict = np.argmax(p_predict, axis=1)
            acc = sklearn.metrics.accuracy_score(y_test, y_predict)
            if len(y_test) == 1:
                auc = 0.0
            elif len(np.unique(y_train)) == 2:
                auc = sklearn.metrics.roc_auc_score(y_test, p_predict[:, 1])
            else:
                auc = 0.0
                for i in range(p_predict.shape[1]):
                    auc += sklearn.metrics.roc_auc_score(y_test == i, p_predict[:, i])
                auc /= p_predict.shape[1]
            kappa = sklearn.metrics.cohen_kappa_score(y_test, y_predict)
            classes = np.unique(y_train)
            np.sort(classes)
            confusion = sklearn.metrics.confusion_matrix(y_test, y_predict, labels=classes)
            res.add('acc', acc)
            # res.add('auc',auc)
            # res.add('kappa',kappa)
            if len(label_names[c]) == 2:
                res.add('sensitivity',
                        float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
                res.add('specificity',
                        float(np.logical_and(y_test != 1, y_predict == y_test).sum()) / (y_test != 1).sum())
            res.add('auc', auc)
            res.add('kappa', kappa)
            res.add('confusion', confusion)

            print('accuracy %f auc %f' % (acc, auc))
            print(confusion)

            if save_train:
                dump(model, model_path)

        print(f'Train instance size-stride: {train_instance_size}-{train_instance_stride}')
        print(f'Test instance size-stride: {test_instance_size}-{test_instance_stride}')
        print(f'Quantiles: {quantiles}')
        print('Cross-validation results')
        res.print_summary()
