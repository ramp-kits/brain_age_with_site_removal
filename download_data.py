# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import warnings
import urllib.request
import pandas as pd
import numpy as np

try:
    PATH_DATA = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")
except NameError:
    PATH_DATA = "data"
os.makedirs(PATH_DATA, exist_ok=True)


def fetch_data(basenames, rootdir, base_url, verbose=1):
    """ Fetch dataset.

    Parameters
    ----------
    files: list of str
        the basename of the files to be fetched.
    rootdir: str
        the destination directory.
    base_url: str
        the base URL where are stored the files to be fetched.

    Returns
    -------
    downloaded list of str
        the paths to the fetched files.
    """
    downloaded = []
    for name in basenames:
        src_filename = os.path.join(base_url, name)
        dst_filename = os.path.join(rootdir, name)
        if not os.path.exists(dst_filename):
            if verbose:
                print("- download: {0}.".format(src_filename))
            try:
                urllib.request.urlretrieve(src_filename, dst_filename)
            except:
                warnings.warn(
                    "Impossible to download the '{0}' file: the dataset may "
                    "be private.".format(name), UserWarning)
        downloaded.append(dst_filename)
    return downloaded


def generate_random_data(rootdir, dtype, n_samples):
    """ Generate random data.

    The test set is composed of data with the same sites as in the
    train set (internal dataset) and data with unseen sites during the
    training (external dataset).
    The first line contains the split index and nans.

    Parameters
    ----------
    rootdir: str
        the data location.
    dtype: str
        the datasset type: 'train' or'test'.
    n_samples: int
        the number of generated samples.

    Returns
    -------
    x_arr: array (n_samples, n_features)
        input data.
    y_arr: array (n_samples, 2)
        target data.
    """
    x_arrs, y_arrs = [], []
    if dtype == "test":
        dtypes = ["internal_test", "external_test"]
    else:
        dtypes = ["internal_train"]
    split_info = []
    for name in dtypes:
        x_arr = np.random.rand(n_samples, 3659572)
        df = pd.DataFrame(data=np.arange(n_samples), columns=["samples"])
        df["age"] = np.random.randint(5, 80, n_samples)
        if name == "external_test":
            df["site"] = 3
        else:
            df["site"] = np.random.randint(0, 2, n_samples)
        y_arr = df[["age", "site"]].values
        split_info.extend([name] * len(y_arr))
        x_arrs.append(x_arr)
        y_arrs.append(y_arr)
    x_arr = np.concatenate(x_arrs, axis=0)
    y_arr = np.concatenate(y_arrs, axis=0)
    df = pd.DataFrame(y_arr, columns=("age", "site"))
    df["split"] = split_info
    np.save(os.path.join(rootdir, dtype + ".npy"), x_arr.astype(np.float32))
    df.to_csv(os.path.join(rootdir, dtype + ".tsv"), sep="\t", index=False)
    return x_arr, y_arr


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--test", action="store_true",
                        help="generate a random test set.")
    args = parser.parse_args()

    if args.test:
        generate_random_data(rootdir=PATH_DATA, dtype="train", n_samples=30)
        generate_random_data(rootdir=PATH_DATA, dtype="test", n_samples=10)
    else:
        fetch_data(basenames=["train.npy", "train.tsv", "test.npy",
                              "test.tsv"],
                   rootdir=PATH_DATA,
                   base_url=("ftp://ftp.cea.fr/pub/unati/share/OpenBHB"),
                   verbose=1)
