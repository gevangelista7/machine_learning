import random
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold


class NestedKFoldAlgoTester:
    def __init__(self, X, y, inner_splits, outer_splits):
        self.result_register_f1 = {}
        self.result_register_precision = {}
        self.result_register_recall = {}
        self.result_register_recall = {}
        self.result_register_f1_neg = {}
        self.result_register_precision_neg = {}
        self.result_register_recall_neg = {}
        self.result_register_accuracy = {}
        self.models = {}

        self.X = X
        self.y = y

        self.y_neg = (1 - y).transform(bool)

        class_dist = list(self.y.value_counts().to_dict().values())
        self.class_dist = list(np.array(class_dist) / np.sum(class_dist))

        self.inner_splits = inner_splits
        self.outer_splits = outer_splits

    def algo_test(self, estimator, param_grid, registry_name):
        if registry_name not in self.result_register_f1.keys():
            self.result_register_f1[registry_name] = []
            self.result_register_precision[registry_name] = []
            self.result_register_recall[registry_name] = []
            self.result_register_precision_neg[registry_name] = []
            self.result_register_recall_neg[registry_name] = []
            self.result_register_f1_neg[registry_name] = []
            self.result_register_accuracy[registry_name] = []

        seed = random.randint(0, 10000)
        run_name = registry_name + "_" + str(seed)
        print("Begin of run: {} \n{}".format(run_name, str(datetime.now())))

        inner_cv = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=seed)
        outer_cv = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=seed)

        clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv)

        print("Begin of f1 cross_val")
        nested_score_f1 = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv,
                                          scoring='accuracy', n_jobs=1)

        print("Begin of precision cross_val")
        nested_score_precision = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv,
                                                 scoring='precision_weighted', n_jobs=1)

        print("Begin of recall cross_val")
        nested_score_recall = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv,
                                              scoring='recall_weighted', n_jobs=1)
        print("Begin of accuracy cross_val")
        nested_score_accuracy = cross_val_score(clf, X=self.X, y=self.y, cv=outer_cv,
                                              scoring='accuracy', n_jobs=1)

        print("Begin of f1_neg cross_val")
        nested_score_f1_neg = cross_val_score(clf, X=self.X, y=self.y_neg, cv=outer_cv,
                                              scoring='f1_weighted', n_jobs=1)

        print("Begin of precision_neg cross_val")
        nested_score_precision_neg = cross_val_score(clf, X=self.X, y=self.y_neg, cv=outer_cv,
                                                 scoring='precision_weighted', n_jobs=1)

        print("Begin of recall_neg cross_val")
        nested_score_recall_neg = cross_val_score(clf, X=self.X, y=self.y_neg, cv=outer_cv,
                                              scoring='recall_weighted', n_jobs=1)

        self.result_register_f1[registry_name].append(nested_score_f1)
        self.result_register_precision[registry_name].append(nested_score_precision)
        self.result_register_recall[registry_name].append(nested_score_recall)
        self.result_register_accuracy[registry_name].append(nested_score_accuracy)
        self.result_register_precision_neg[registry_name].append(nested_score_precision_neg)
        self.result_register_recall_neg[registry_name].append(nested_score_recall_neg)
        self.result_register_f1_neg[registry_name].append(nested_score_f1_neg)

        print("End of run: {} \nScore f1: {}".format(run_name, nested_score_precision_neg.mean()))
        # print("End of run: {} \nScore f1: {}".format(run_name, nested_score_accuracy.mean()))