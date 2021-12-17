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

    Parameters
    ----------
    rootdir: str
        the data location.
    dtype: str
        the datasset type: 'train', 'test', or 'private_test'.
    n_samples: int
        the number of generated samples.

    Returns
    -------
    x_arr: array (n_samples, n_features)
        input data.
    y_arr: array (n_samples, )
        target data.
    """
    x_arr = np.random.rand(n_samples, 2348836)
    df = pd.DataFrame(data=range(n_samples), columns=["samples"])
    df["age"] = np.random.randint(5, 80, n_samples)
    if dtype == "private_test":
        df["site"] = ""
    else:
        df["site"] = np.random.randint(0, 2, n_samples)
    np.save(os.path.join(rootdir, dtype + ".npy"), x_arr.astype(np.float32))
    df.to_csv(os.path.join(rootdir, dtype + ".tsv"), sep="\t", index=False)
    y_arr = df[["age", "site"]].values
    return x_arr, y_arr


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--test", action="store_true",
                        help="generate a random test set.")
    args = parser.parse_args()

    if args.test:
        generate_random_data(rootdir=PATH_DATA, dtype="train", n_samples=20)
        generate_random_data(rootdir=PATH_DATA, dtype="test", n_samples=10)
        generate_random_data(rootdir=PATH_DATA, dtype="private_test",
                             n_samples=10)
    else:
        fetch_data(basenames=[
                        "train.npy", "train.tsv", "test.npy", "test.tsv",
                        "private_test.npy", "private_test.tsv"],
                   rootdir=PATH_DATA,
                   base_url=("ftp://ftp.cea.fr/pub/unati/share/OpenBHB"),
                   verbose=1)
