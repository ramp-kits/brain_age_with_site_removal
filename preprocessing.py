# -*- coding: utf-8 -*-
##########################################################################
# Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
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
                     ("private_test", private_test_df)):
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
        if key == "vbm":
            arr = arr[..., 0]
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


if __name__ == "__main__":

    organize_data(rootdir="/neurospin/hc")
