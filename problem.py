# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
This parametrizes the setup using building blocks from RAMP workflow.
"""

import os
import pandas as pd
import numpy as np
import rampwf as rw
from rampwf.utils.pretty_print import print_title
from sklearn.base import BaseEstimator
from sklearn.utils import _safe_indexing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, r2_score, balanced_accuracy_score)


class DeepDebiasingEstimator(rw.workflows.SKLearnPipeline):
    """ Wrapper to convert a scikit-learn estimator into a Deep Learning
    Debiasing RAMP workflow that use a public and a private dataset as well
    as two regressors (age & site) on latent space network features.

    Notes
    -----
    The training is not performed on the server side. Weights are required at
    the submission time and fold indices are tracked at training time.
    """
    def __init__(self, memory, filename="estimator.py",
                 additional_filenames=None, max_n_features=10000):
        super().__init__(filename, additional_filenames)
        self.memory = memory
        self.max_n_features = max_n_features

    def train_submission(self, module_path, X, y, train_idx=None):
        """ Train the estimator of a given submission.

        Parameters
        ----------
        module_path : str
            The path to the submission where `filename` is located.
        X : array-like, dataframe of shape (n_samples, n_features)
            The data matrix.
        y : array-like of shape (n_samples, 2)
            The target vector: age, site.
        train_idx : array-like of shape (n_training_samples,), default None
            The training indices. By default, the full dataset will be used
            to train the model. If an array is provided, `X` and `y` will be
            subsampled using these indices.

        Returns
        -------
        estimators : estimator objects
            Scikit-learn models fitted on (`X`, `y`).
        """
        print_title("DeepDebiasingEstimator...")
        _train_idx = (slice(None, None, None) if train_idx is None
                      else train_idx)
        submission_module = rw.utils.importing.import_module_from_source(
            os.path.join(module_path, self.filename),
            os.path.splitext(self.filename)[0],
            sanitize=True
        )
        estimator = submission_module.get_estimator()
        y_train = _safe_indexing(y, _train_idx)
        y_age, y_site = y_train.T
        y_age = y_age.astype(float)
        y_site = y_site.astype(int)
        print("- X:", X.shape, len(_train_idx))
        print("- y:", y_train.shape)
        print("- ages:", y_age.shape)
        print("- sites:", y_site.shape)
        print_title("Restoring weights...")
        features_estimator = estimator.fit(None, None)
        print_title("Estimates features...")
        for item in estimator:
            if hasattr(item, "indices"):
                item.indices = train_idx
        features = features_estimator.predict(X)
        for item in estimator:
            if hasattr(item, "indices"):
                item.indices = None
        if features.shape[1] > self.max_n_features:
            raise ValueError(
                "You reached the maximum authorized size of the feature space "
                "({0}).".format(self.max_n_features))
        print("- features:", features.shape)
        print_title("Estimates sites from features...")
        site_estimator = SiteEstimator()
        site_estimator.fit(features, y_site)
        print_title("Estimates age from features...")
        age_estimator = AgeEstimator()
        age_estimator.fit(features, y_age)
        if "y_pred" in self.memory:
            self.memory["y_pred"] = None
        return features_estimator, site_estimator, age_estimator

    def test_submission(self, estimators_fitted, X):
        """Predict using a fitted estimator.

        Parameters
        ----------
        estimators_fitted : estimator objects
            Fitted scikit-learn estimators.
        X : array-like, dataframe of shape (n_samples, n_features)
            The test data set.

        Returns
        -------
        pred : ndarray of shape (n_samples, n_classes) or (n_samples)
        """
        print_title("Testing...")
        print("- X:", X.shape)
        features_estimator, site_estimator, age_estimator = estimators_fitted
        features = features_estimator.predict(X)
        y_site_pred = site_estimator.predict(features)
        y_age_pred = age_estimator.predict(features)
        print("- features:", features.shape)
        print("- y site:", y_site_pred.shape)
        print("- y age:", y_age_pred.shape)
        if len(self.memory) == 0:
            # TODO: set absolute path to the data in prod.
            private_X, private_y = get_private_test_data()
            private_y_age = private_y[:, 0].astype(float)
            self.memory.update({
                "X": private_X,
                "y_true": private_y_age,
                "y_pred": None
            })
        if self.memory["y_pred"] is None:
            private_features = features_estimator.predict(self.memory["X"])
            y_age_private_pred = age_estimator.predict(private_features)
            self.memory["y_pred"] = y_age_private_pred
            print("- features [private]:", features.shape)
            print("- y age [private]:", y_age_private_pred.shape)
        return np.concatenate([
            y_age_pred.reshape(-1, 1), y_site_pred], axis=1)


class SiteEstimator(BaseEstimator):
    """ Define the site estimator on latent space network features.
    """
    def __init__(self):
        self.site_estimator = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=150), cv=5,
            param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy")

    def fit(self, X, y):
        self.site_estimator.fit(X, y)

    def predict(self, X):
        y_pred = self.site_estimator.predict_proba(X)
        return y_pred


class AgeEstimator(BaseEstimator):
    """ Define the age estimator on latent space network features.
    """
    def __init__(self):
        self.age_estimator = GridSearchCV(
            Ridge(), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2")

    def fit(self, X, y):
        self.age_estimator.fit(X, y)

    def predict(self, X):
        y_pred = self.age_estimator.predict(X)
        return y_pred


class R2(rw.score_types.BaseScoreType):
    """ Compute coefficient of determination usually denoted as R2.
    """
    is_lower_the_better = False
    minimum = -float("inf")
    maximum = 1

    def __init__(self, name="mae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return r2_score(y_true, y_pred)


class MAE(rw.score_types.BaseScoreType):
    """ Compute mean absolute error, a risk metric corresponding to the
    expected value of the absolute error loss or l1-norm loss.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="mae", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)


class ExtMAE(rw.score_types.BaseScoreType):
    """ Compute mean absolute error on the private external test set.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, memory, name="mae", precision=2):
        self.memory = memory
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return mean_absolute_error(self.memory["y_true"],
                                   self.memory["y_pred"])


class BACC(rw.score_types.classifier_base.ClassifierBaseScoreType):
    """ Compute balanced accuracy, which avoids inflated performance
    estimates on imbalanced datasets.
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="bacc", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        return balanced_accuracy_score(y_true_label_index, y_pred_label_index)


class DeepDebiasingMetric(rw.score_types.BaseScoreType):
    """ Compute the final metric as descibed in the paper.
    """
    is_lower_the_better = False
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, score_types, n_sites, memory, name="combined",
                 precision=2):
        self.name = name
        self.score_types = score_types
        self.n_sites = n_sites
        self.precision = precision
        self.memory = memory
        self.score_type_mae_private_age = MAE(
            name="mae_private_age", precision=3)

    def score_function(self, ground_truths_combined, predictions_combined,
                       valid_indexes=None):
        scores = {}
        scores[self.score_type_mae_private_age.name] = (
            self.score_type_mae_private_age(self.memory["y_true"],
                                            self.memory["y_pred"]))
        for score_type, ground_truths, predictions in zip(
                self.score_types,
                ground_truths_combined.predictions_list,
                predictions_combined.predictions_list):
            scores[score_type.name] = score_type.score_function(
                ground_truths, predictions, valid_indexes)
        metric = (
            scores["mae_private_age"] * scores["bacc_site"] +
            (1. / self.n_sites) * scores["mae_age"])
        return metric

    def __call__(self, y_true, y_pred):
        raise ValueError("Combined score has no deep score function.")


def get_cv(X, y):
    """ Get N folds cross validation indices.
    """
    cv_train = KFold(n_splits=2, shuffle=True, random_state=0)
    folds = []
    for cnt, (train_idx, test_idx) in enumerate(cv_train.split(X, y)):
        train_idx = np.insert(train_idx, 0, cnt)
        folds.append((train_idx, test_idx))
    return folds


def _read_data(path, dataset):
    """ Read data.

    Parameters
    ----------
    path: str
        the data location.
    dataset: str
        'train', 'test' or 'private_test'.

    Returns
    -------
    x_arr: array (n_samples, n_features)
        input data.
    y_arr: array (n_samples, )
        target data.
    """
    if dataset == "private_test" and not os.path.isfile(os.path.join(
            path, "data", dataset + ".tsv")):
        dataset = "test"
    df = pd.read_csv(os.path.join(path, "data", dataset + ".tsv"), sep="\t")
    y_arr = df[["age", "site"]].values
    x_arr = np.load(os.path.join(path, "data", dataset + ".npy"),
                    mmap_mode="r")
    return x_arr, y_arr


def get_train_data(path="."):
    """ Get openBHB public train set.
    """
    return _read_data(path, "train")


def get_test_data(path="."):
    """ Get openBHB public test set.
    """
    return _read_data(path, "test")


def get_private_test_data(path="."):
    """ Get privateBHB test set.

    This private test set is unable during local executions of the code.
    The set is used when computing the combined loss. Locally this set is
    replaced by the public test set, and the combined loss may not be relevant.
    """
    return _read_data(path, "private_test")


problem_title = (
    "Brain age prediction and debiasing with site-effect removal in MRI "
    "through representation learning.")
# _, y = get_train_data()
# _prediction_site_names = sorted(np.unique(y[:, 1]))
# TODO: switch labels manually if using the test set in prod
# _prediction_site_names = [0, 1]
_prediction_site_names = list(range(64))
_target_column_names = ["age", "site"]
private_mae_memory = {}
Predictions_age = rw.prediction_types.make_regression(
    label_names=[_target_column_names[0]])
Predictions_site = rw.prediction_types.make_multiclass(
    label_names=_prediction_site_names)
Predictions = rw.prediction_types.make_combined(
    [Predictions_age, Predictions_site])
score_type_r2_age = R2(name="r2_age", precision=3)
score_type_mae_age = MAE(name="mae_age", precision=3)
score_type_rmse_age = rw.score_types.RMSE(name="rmse_age", precision=3)
score_type_acc_site = rw.score_types.Accuracy(name="acc_site", precision=3)
score_type_bacc_site = BACC(name="bacc_site", precision=3)
score_type_ext_mae_age = ExtMAE(memory=private_mae_memory, name="ext_mae_age",
                                precision=3)
score_types = [
    DeepDebiasingMetric(
        name="challenge_metric", precision=3,
        n_sites=len(_prediction_site_names), memory=private_mae_memory,
        score_types=[score_type_mae_age, score_type_bacc_site]),
    rw.score_types.MakeCombined(score_type=score_type_r2_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_mae_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_rmse_age, index=0),
    rw.score_types.MakeCombined(score_type=score_type_acc_site, index=1),
    rw.score_types.MakeCombined(score_type=score_type_bacc_site, index=1),
    rw.score_types.MakeCombined(score_type=score_type_ext_mae_age, index=0)
]
workflow = DeepDebiasingEstimator(
    memory=private_mae_memory, filename="estimator.py",
    additional_filenames=["weights.pth", "metadata.pkl"])
