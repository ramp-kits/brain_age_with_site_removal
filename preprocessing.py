# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import json
from collections import OrderedDict
import pandas as pd
import numpy as np
import nibabel
from nilearn.masking import apply_mask


MODALITITES = OrderedDict([
    ("vbm", {
        "pattern": "sub-{0}_preproc-cat12vbm_desc-gm_T1w.npy",
        "shape": (-1, 1, 121, 145, 121),
        "size": 519945}),
    ("quasiraw", {
        "pattern": "sub-{0}_preproc-quasiraw_T1w.npy",
        "shape": (-1, 1, 182, 218, 182),
        "size": 1827095}),
    ("xhemi", {
        "pattern": "sub-{0}_preproc-freesurfer_desc-xhemi_T1w.npy",
        "shape": (-1, 8, 163842),
        "size": 1310736}),
    ("vbm_roi", {
        "pattern": "sub-{0}_preproc-cat12vbm_desc-gm_ROI.npy",
        "shape": (-1, 1, 284),
        "size": 284}),
    ("desikan_roi", {
        "pattern": "sub-{0}_preproc-freesurfer_desc-desikan_ROI.npy",
        "shape": (-1, 7, 68),
        "size": 476}),
    ("destrieux_roi", {
        "pattern": "sub-{0}_preproc-freesurfer_desc-destrieux_ROI.npy",
        "shape": (-1, 7, 148),
        "size": 1036})
])

MASKS = {
    "vbm": {
        "basename": "cat12vbm_space-MNI152_desc-gm_TPM.nii.gz",
        "thr": 0.05},
    "quasiraw": {
        "basename": "quasiraw_space-MNI152_desc-brain_T1w.nii.gz",
        "thr": 0}
}


def organize_data(rootdir):
    """ Organize multi-model dataset: flatten and concatenate all data.

    Parameters
    ----------
    rootdir: str
        root directory.
    """
    opendir = os.path.join(rootdir, "openBHB")
    privatedir = os.path.join(rootdir, "privateBHB")
    opendatadir = os.path.join(opendir, "data")
    privatedatadir = os.path.join(privatedir, "data")
    resourcedir = os.path.join(opendir, "resource")
    open_df = pd.read_csv(os.path.join(
        opendir, "participants.tsv"), sep="\t")
    private_df = pd.read_csv(os.path.join(
        privatedir, "participants.tsv"), sep="\t")
    train_df = pd.read_csv(os.path.join(
        opendir, "train_site_labels_v1.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(
        opendir, "test_site_labels_v1.tsv"), sep="\t")
    private_test_df = pd.read_csv(os.path.join(
        privatedir, "test_site_labels_v1.tsv"), sep="\t")
    train_df = _intersect(train_df, open_df, opendatadir)
    test_df = _intersect(test_df, open_df, opendatadir)
    private_test_df = _intersect(private_test_df, private_df, privatedatadir)
    private_test_df["site"] = ""
    for name, df in (("train", train_df), ("test", test_df),
                     ("private_external_test", private_test_df)):
        print("-", name)
        print(df)
        data = _load(df, resourcedir)
        df = df[["participant_id", "age", "site"]]
        df.to_csv(os.path.join(rootdir, "{0}.tsv".format(name)),
                  sep="\t", index=False)
        np.save(os.path.join(rootdir, "{0}.npy".format(name)), data)


def _intersect(df, meta_df, datadir):
    data = {}
    for cnt, row in df.iterrows():
        sid = row.participant_id.item()
        age = meta_df.loc[meta_df["participant_id"] == sid].age.item()
        data.setdefault("age", []).append(age)
        for key, info in MODALITITES.items():
            path = os.path.join(datadir, info["pattern"].format(sid))
            data.setdefault(key, []).append(path)
    for key, val in data.items():
        df[key] = val
    df.rename(columns={"siteXacq": "site"}, inplace=True)
    return df


def _load(df, resourcedir):
    affine = np.eye(4)
    masks = dict((key, os.path.join(resourcedir, val["basename"]))
                 for key, val in MASKS.items())
    for key in masks:
        arr = nibabel.load(masks[key]).get_fdata()
        thr = MASKS[key]["thr"]
        arr[arr <= thr] = 0
        arr[arr > thr] = 1
        masks[key] = nibabel.Nifti1Image(arr.astype(int), affine)
    data = []
    n_subjects = len(df)
    for key in MODALITITES:
        _cdata = []
        print("-", key)
        for cnt, path in enumerate(df[key].values):
            if cnt % 100 == 0:
                print("--", cnt, "/", n_subjects, "--")
            arr = np.load(path)
            if key in ("vbm", "quasiraw"):
                im = nibabel.Nifti1Image(arr.squeeze(), affine)
                arr = apply_mask(im, masks[key])
                arr = np.expand_dims(arr, axis=0)
            else:
                arr = arr.reshape(1, -1)
            _cdata.append(arr)
        _cdata = np.concatenate(_cdata, axis=0)
        print("- bloc shape:", _cdata.shape)
        data.append(_cdata.astype(np.float32))
    data = np.concatenate(data, axis=1)
    print("- datasset shape:", data.shape)
    return data


def concat_datasets(rootdir):
    """ Format test set.

    The test set is composed of data with the same sites as in the
    train set (internal dataset) and data with unseen sites during the
    training (external dataset).

    Parameters
    ----------
    rootdir: str
        root directory.
    """
    internal_test_file = os.path.join(rootdir, "internal_test")
    external_test_file = os.path.join(rootdir, "external_test")
    name = "test"
    locations = (internal_test_file, external_test_file)
    y_dfs = [pd.read_csv(path + ".tsv", sep="\t") for path in locations]
    for df, loc in zip(y_dfs, locations):
        dtype = os.path.basename(loc)
        df["split"].replace(name, dtype, inplace=True)
    y_df = pd.concat(y_dfs)
    y_df.to_csv(os.path.join(rootdir, name + ".tsv"), sep="\t",
                index=False)
    x_arrs = [np.load(path + ".npy", mmap_mode="r") for path in locations]
    x_arr = np.concatenate(x_arrs, axis=0)
    np.save(os.path.join(rootdir, name + ".npy"), x_arr)


def compile_resources(rootdir):
    """ Compile all resources in one file.

    Parameters
    ----------
    rootdir: str
        root directory.
    """
    data = {
        "vbm": {
            "shape": (-1, 1, 121, 145, 121),
            "features": None,
            "channels": ["GM"]},
        "quasiraw": {
            "shape": (-1, 1, 182, 218, 182),
            "features": None,
            "channels": ["T1w"]},
        "vbm_roi": {
            "shape": (-1, 1, 284),
            "features": np.loadtxt(os.path.join(
                rootdir, "cat12vbm_labels.txt"), dtype=str),
            "channels": ["GM"]},
        "desikan_roi": {
            "shape": (-1, 7, 68),
            "features": np.loadtxt(os.path.join(
                rootdir, "freesurfer_atlas-desikan_labels.txt"), dtype=str),
            "channels": np.loadtxt(os.path.join(
                rootdir, "freesurfer_channels.txt"), dtype=str)},
        "destrieux_roi": {
            "shape": (-1, 7, 148),
            "features": np.loadtxt(os.path.join(
                rootdir, "freesurfer_atlas-destrieux_labels.txt"), dtype=str),
            "channels": np.loadtxt(os.path.join(
                rootdir, "freesurfer_channels.txt"), dtype=str)},
        "xhemi": {
            "shape": (-1, 8, 163842),
            "features": None,
            "channels": np.loadtxt(os.path.join(
                rootdir, "freesurfer_xhemi_channels.txt"), dtype=str)}
    }

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Not serializable")

    with open(os.path.join(rootdir, "resources.json"), "wt") as of:
        json.dump(data, of, indent=2, default=default)


def convert_split(rootdir):
    """ Convert the split and use subject index in table rather than subject
    id.

    Parameters
    ----------
    rootdir: str
        root directory.
    """
    split_file = os.path.join(rootdir, "cv_splits.json")
    with open(split_file, "rt") as of:
        split_data = json.load(of)
    desc_file = os.path.join(rootdir, "train.tsv")
    df = pd.read_csv(desc_file, sep="\t")
    print(df)
    splits = {}
    for fold_name, sets in split_data.items():
        splits[fold_name] = {}
        for set_name, subjects in sets.items():
            print(fold_name, set_name, len(subjects))
            for sid in subjects:
                index = df[df["participant_id"] == sid].index.tolist()
                assert len(index) == 1, sid
                splits[fold_name].setdefault(set_name, []).append(index[0])
    split_file = os.path.join(rootdir, "cv_splits_indices.json")
    with open(split_file, "wt") as of:
        json.dump(splits, of, indent=2)


if __name__ == "__main__":

    # organize_data(rootdir="/neurospin/hc")
    # concat_datasets(
    #     rootdir="/neurospin/hc/challengeBHB/public_data_challenge")
    # concat_datasets(
    #     rootdir="/neurospin/hc/challengeBHB/private_data_challenge")
    # compile_resources(
    #     rootdir="/neurospin/hc/openBHB/resource")
    # convert_split(
    #     rootdir="/neurospin/hc/challengeBHB/public_data_challenge")
    convert_split(
        rootdir="/neurospin/hc/challengeBHB/private_data_challenge")
