import os
import sys
import argparse
import warnings
import numpy as np
import sklearn.metrics
import sksurv.metrics
from joblib import dump, load

import util
from linear_classifier import LinearClassifier
from sil import SIL


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

            if c_index:
                self.print_summary('c_index_continuous')
                self.print_summary('c_index_binary')
                # print(f"C-index (continuous): {self.res['c_index_continuous']}")
                # print(f"C-index (binary): {self.res['c_index_binary']}")
            # for metric in sorted(self.res.keys()):
            # for metric in self.res.keys():
            #     if metric != 'confusion':
            #         self.print_summary(metric)
            self.print_summary('confusion')
            return
        if metric != 'confusion':
            mean = np.mean(self.res[metric])
            std = np.std(self.res[metric])
            ste = std / np.sqrt(len(self.res[metric]))
            print('%s %f %f %f' % (metric, mean, std, ste))
        # def print_summary(self, metric=None):
        #     if metric is None:
        #         print(f"""
        #         Accuracy Sensitivity Specificity AUC
        #         {np.mean(self.res['acc'])},{np.mean(self.res['sensitivity'])},{np.mean(self.res['specificity'])},{np.mean(self.res['auc'])}
        #         """)
        #         if c_index:
        #             self.print_summary('c_index_continuous')
        #             self.print_summary('c_index_binary')
        #             # print(f"C-index (continuous): {self.res['c_index_continuous']}")
        #             # print(f"C-index (binary): {self.res['c_index_binary']}")
        #         # for metric in sorted(self.res.keys()):
        #         # for metric in self.res.keys():
        #         #     if metric != 'confusion':
        #         #         self.print_summary(metric)
        #         self.print_summary('confusion')
        #         return
        #     if metric != 'confusion':
        #         mean = np.mean(self.res[metric])
        #         std = np.std(self.res[metric])
        #         ste = std / np.sqrt(len(self.res[metric]) - 1)
        #         print('%s %f %f %f' % (metric, mean, std, ste))

        else:
            print('confusion')
            print(('%s ' * len(self.label_names)) % tuple(self.label_names))
            print(sum(self.res['confusion']))
            print('Negative / Positive')
            print(np.sum(sum(self.res['confusion']), axis=1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test MI classifier.')
    parser.add_argument('--out_dir', '-o', required=True, help='output directory')
    parser.add_argument('--model', '-m', required=True, help='CNN model')
    parser.add_argument('--layer', '-l', required=True, help='CNN layer')
    parser.add_argument('--instance-size', help='instance size')
    parser.add_argument('--instance-stride', help='instance stride')
    parser.add_argument('--pool-size', '-p', help='mean pooling size')
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all')
    parser.add_argument('--calibrate', action='store_true', help='calibrate classifier')
    parser.add_argument('--metric',
                        help='metric to optimize during parameter search (accuracy, balanced_accuracy, roc_auc); default: accuracy')
    parser.add_argument('--classifier', '-c', help='classifier (logistic, svm, dwd); default: svm')
    parser.add_argument('--kernel', help='SVM kernel; default: linear')
    parser.add_argument('--mi',
                        help='MI type (none, median, max, quantile, quantile_mean, quantile_pca, quantile_mean_pca, quantile_opc); default: none (compute mean across images)')
    parser.add_argument('--agg', help='Aggregate type (svm, dwd); default: svm')
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16')
    parser.add_argument('--sample-weight', help='Weight samples by classification category and this one')
    parser.add_argument('--group', help='Class groups for reporting results')
    parser.add_argument('--cv-fold-files', help='cross-validation fold files')
    parser.add_argument('--cv-folds', help='cross-validation folds')
    parser.add_argument('--cv-lno', help='cross-validation leave n out')
    parser.add_argument('--random-state', help='random state for splitting datasets')
    parser.add_argument('--save-first', action='store_true', help='save first trained classifier and datasets')
    parser.add_argument('--save-idx', action='store_true', help='save indices of first trained classifier and datasets')
    parser.add_argument('--c-index', action='store_true', help='c-index for survival analysis')
    parser.add_argument('--n-jobs', help='number of parallel threads')
    parser.add_argument('--n-components', help='number of principal components')
    parser.add_argument('--save-train', action='store_false', help='save trained classifier')
    parser.add_argument('--load-train', action='store_false', help='load trained classifier')
    args = parser.parse_args()
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    model_name = args.model
    layer = args.layer
    instance_size = args.instance_size
    instance_stride = args.instance_stride
    pool_size = args.pool_size
    categories = args.cat
    metric = args.metric
    calibrate = args.calibrate
    classifier = args.classifier
    kernel = args.kernel
    mi_type = args.mi
    agg_type = args.agg
    quantiles = args.quantiles
    sample_weight = args.sample_weight
    group = args.group
    cv_fold_files = args.cv_fold_files
    cv_folds = args.cv_folds
    cv_lno = args.cv_lno
    random_state = args.random_state
    save_first = args.save_first
    save_idx = args.save_idx
    c_index = args.c_index
    n_jobs = args.n_jobs
    n_components = args.n_components
    save_train = args.save_train
    load_train = args.load_train

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

    if save_first is None:
        save_first = False
    else:
        save_first = bool(save_first)

    if save_idx is None:
        save_idx = False
    else:
        save_idx = bool(save_idx)

    if c_index is None:
        c_index = False
    else:
        c_index = bool(c_index)

    if n_jobs is not None:
        n_jobs = int(n_jobs)

    if random_state is None:
        random_state = 111
    else:
        random_state = int(random_state)

    if n_components is not None:
        n_components = int(n_components)

    if save_train is None:
        save_train = True
    else:
        save_train = bool(save_train)

    if load_train is None:
        load_train = True
    else:
        load_train = bool(load_train)

    # load filenames and labels
    sample_images = util.load_sample_images(out_dir)
    samples, cats, labels = util.load_labels(out_dir)

    # TODO: only keep labels with corresponding feats

    if sample_weight is not None:
        # get labels for sample_weight category
        c = np.where(cats == sample_weight)[0][0]
        ln = np.unique([l[c] for l in labels])
        ln.sort()
        ln = list(ln)
        if '' in ln:
            del ln[ln.index('')]
        label_names_sw = ln
        labels_sw = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])
    if group is not None:
        # get labels for group category
        if group == sample_weight:
            label_names_group = label_names_sw
            labels_group = labels_sw
        else:
            c = np.where(cats == group)[0][0]
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            if '' in ln:
                del ln[ln.index('')]
            label_names_group = ln
            labels_group = np.array([ln.index(l) if l in ln else -1 for l in labels[:, c]])
    if categories is None:
        # get labels for all categories
        label_names = []
        new_labels = np.zeros(labels.shape, dtype='int')
        for c, cat in enumerate(cats):
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            label_names.append(ln)
            new_labels[:, c] = [ln.index(l) for l in labels[:, c]]
        labels = new_labels
    else:
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
        if mi_type is None or mi_type.lower() == 'none':
            if len(feats[sample].shape) > 1:
                feats[sample] = feats[sample].mean(axis=0)

    # build train/test sets
    if cv_fold_files is not None:
        idx_train_test = util.load_cv_files(out_dir, samples, cv_fold_files)
    elif cv_folds is not None or cv_lno is not None:
        if cv_folds is not None:
            cv_folds = int(cv_folds)
        else:
            cv_lno = int(cv_lno)
            if cv_folds is None:
                cv_folds = len(samples) // cv_lno
        idx = np.arange(len(samples))

    else:
        print('Error: train/test split not specified')
        sys.exit(1)

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

    for c, cat_name in enumerate(cats):
        idx = [i for i in idx if (labels[i, c] != -1) & (samples[i] in feats)]
        labels_subset = labels[idx]

        if len(label_names) == 1:
            if cv_lno == 1:
                skf = sklearn.model_selection.LeaveOneOut()
            else:
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

        print(cat_name)
        res = ResultsReport(label_names[c])
        for f, (idx_train, idx_test) in enumerate(idx_train_test):
            print('Fold ' + str(f + 1) + '/' + str(len(idx_train_test)))
            # idx_train = [i for i in idx_train if (labels[i, c] != -1) & (samples[i] in feats)]
            # idx_test = [i for i in idx_test if (labels[i, c] != -1) & (samples[i] in feats)]
            # idx_train = idx_train[np.where(labels[idx_train,c] != -1)[0]]
            # idx_test = idx_test[np.where(labels[idx_test,c]!=-1)[0]]
            idx_train = np.array(idx)[idx_train]
            idx_test = np.array(idx)[idx_test]
            X_train = [feats[samples[i]] for i in idx_train]
            y_train = labels[idx_train, c]
            X_test = [feats[samples[i]] for i in idx_test]
            y_test = labels[idx_test, c]

            # load trained classifier
            if mi_type is None:
                model_path = out_dir + '_' + 'mean' + '_' + classifier + '_' + cat_name + '_i' \
                             + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) \
                             + '_fold' + str(f)
            else:
                model_path = out_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' \
                             + str(instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) \
                             + '_fold' + str(f)

            if sample_weight is not None:
                # figure out sample weights
                print('Weighting by ' + sample_weight)
                # discard samples missing a label for sample_weight category
                idx_train = idx_train[np.where(labels_sw[idx_train] != -1)[0]]
                X_train = [feats[samples[i]] for i in idx_train if samples[i] in feats]

                y_train = labels[idx_train, c]
                y_sw = y_train + len(label_names[c]) * labels_sw[idx_train]

                uniq = np.unique(y_sw).tolist()
                counts = np.array([(y_sw == l).sum() for l in uniq])
                counts = counts.sum().astype(float) / (counts * len(counts))
                sw = np.array([counts[uniq.index(y)] for y in y_sw])

                # if load_train and os.path.exists(model_path):
                #     model = load(model_path)
                if mi_type is None:
                    model = LinearClassifier(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)
                elif mi_type in ['median', 'max']:
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)
                elif 'quantile' in mi_type:
                    if quantiles is not None:
                        options['quantiles'] = int(quantiles)
                    model = SIL(n_jobs=n_jobs, **options)
                    model.fit(X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw)

            else:
                # if load_train and os.path.exists(model_path):
                #     model = load(model_path)
                if mi_type is None:
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
            if c_index:
                c_index_continuous = sksurv.metrics.concordance_index_censored(
                    [True] * len(y_test), y_test, -p_predict[:, 1]
                )[0]
                c_index_binary = sksurv.metrics.concordance_index_censored(
                    [True] * len(y_test), y_test, -y_predict
                )[0]
            kappa = sklearn.metrics.cohen_kappa_score(y_test, y_predict)
            classes = np.unique(y_train)
            np.sort(classes)
            confusion = sklearn.metrics.confusion_matrix(y_test, y_predict, labels=classes)
            res.add('acc', acc)
            # res.add('auc', auc)
            # res.add('kappa', kappa)
            if len(label_names[c]) == 2:
                res.add('sensitivity',
                        float(np.logical_and(y_test == 1, y_predict == y_test).sum()) / (y_test == 1).sum())
                res.add('specificity',
                        float(np.logical_and(y_test != 1, y_predict == y_test).sum()) / (y_test != 1).sum())
            res.add('auc', auc)
            if c_index:
                res.add('c_index_continuous', c_index_continuous)
                res.add('c_index_binary', c_index_binary)
            res.add('kappa', kappa)
            res.add('confusion', confusion)

            print('accuracy %f auc %f' % (acc, auc))
            print(confusion)

            if save_train:
                dump(model, model_path)

            if save_first:
                # model_path = out_dir + '_' + classifier + '_' + cat_name
                first_model_path = out_dir + '_' + mi_type + '_' + classifier + '_' + cat_name + '_i' + str(
                    instance_size) + '-' + str(instance_stride) + '_q' + str(quantiles) + '_first'
                data_path = out_dir + '_data_' + cat_name + '_random_state_' + str(random_state)

                dump(model, first_model_path)
                if save_idx:
                    dump([(np.array(samples)[idx_train], X_train, y_train),
                          (np.array(samples)[idx_test], X_test, y_test)],
                         data_path)
                else:
                    dump([(X_train, y_train), (X_test, y_test)], data_path)
                break

            if group is not None:
                # within group class metrics
                l_group = labels_group[idx_test]
                uniq = np.unique(l_group)
                uniq.sort()
                for u in uniq:
                    if u == -1:
                        continue
                    idx = (l_group == u)

                    group_name = '(%s=%s)' % (group, label_names_group[u])
                    res.add('accuracy ' + group_name, sklearn.metrics.accuracy_score(y_test[idx], y_predict[idx]))
                    if len(label_names[c]) == 2:
                        res.add('sensitivity ' + group_name,
                                float(np.logical_and(y_test[idx] == 1, y_predict[idx] == y_test[idx]).sum()) / (
                                        y_test[idx] == 1).sum())
                        res.add('specificity ' + group_name,
                                float(np.logical_and(y_test[idx] != 1, y_predict[idx] == y_test[idx]).sum()) / (
                                        y_test[idx] != 1).sum())
                    if len(np.unique(y_train)) == 2:
                        if (y_test[idx] == 0).sum() == 0 or (y_test[idx] == 1).sum() == 0:
                            auc = 0
                        else:
                            auc = sklearn.metrics.roc_auc_score(y_test[idx], p_predict[idx, 1])
                    else:
                        auc = 0.0
                        for i in range(p_predict.shape[1]):
                            auc += sklearn.metrics.roc_auc_score(y_test[idx] == i, p_predict[idx, i])
                        auc /= p_predict.shape[1]
                    res.add('auc ' + group_name, auc)
                    res.add('kappa ' + group_name, sklearn.metrics.cohen_kappa_score(y_test[idx], y_predict[idx]))
                    # if len(label_names[c]) == 2:
                    #     res.add('sensitivity '+group_name,float( np.logical_and(y_test[idx]==1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]==1).sum() )
                    #     res.add('specificity '+group_name,float( np.logical_and(y_test[idx]!=1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]!=1).sum() )

        print(f'Instance size-stride: {instance_size}-{instance_stride}')
        print(f'Quantiles: {quantiles}')
        print('Cross-validation results')
        res.print_summary()
