# -*- coding: utf-8 -*-
import time
import nibabel as nib
import numpy as np


def read_nii(volume_path):
    nii_obj= nib.load(volume_path)
    nii_data= nii_obj.get_fdata()
    return nii_data, nii_obj

def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(bool)
    im2 = np.asarray(y_pred).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def jaccard_index(A, B):
    '''Compute Jaccard index (IoU) between two segmentation masks.
    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: Jaccard index
    '''

    both = np.logical_and(A, B)
    either = np.logical_or(A, B)
    ji = int(np.sum(both)) / int(np.sum(either))

    return ji


def dice_coef_multiclass(y_true, y_pred, class_id):
    y_true_class = (y_true == class_id)
    y_pred_class = (y_pred == class_id)
    return dice_coef(y_true_class, y_pred_class)

def jaccard_index_multiclass(y_true, y_pred, class_id):
    y_true_class = (y_true == class_id)
    y_pred_class = (y_pred == class_id)
    return jaccard_index(y_true_class, y_pred_class)


def format_inference_time(inference_time):
  if inference_time < 60:
    sec = int(inference_time)
    msec = (inference_time % 1) * 1000
    return f"{sec} seconds:{msec:.2f} milliseconds" 
  else:
    min = int(inference_time / 60)
    sec = inference_time % 60
    msec = (inference_time % 1) * 1000
    return f"{min} minutes:{sec:.2f}.{msec:.2f} sec"


def average_inference_time(total_inference_time, num_volumes):
    average_inference_time = total_inference_time / num_volumes
    return average_inference_time

