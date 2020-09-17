import numpy as np
from scipy.spatial.distance import directed_hausdorff, jaccard


def hausdorff_distance(y_true, y_pred):
    """
    hausdorff distance for binary 3D segmentation

    Args :
        Inputs must be ndarray of 0 or 1 (binary)
        :param y_true: true label image of shape (z, y, x) or (x, y, z)
        :param y_pred: pred label image of shape (z, y, x) or (x, y, z)


    :return: hausdorff distance
    """
    true_vol_idx = np.where(y_true)
    pred_vol_idx = np.where(y_pred)

    return max(directed_hausdorff(true_vol_idx, pred_vol_idx)[0], directed_hausdorff(pred_vol_idx, true_vol_idx)[0])


def AVD(y_true, y_pred):
    """
    Average volume difference

    Args :
        :param y_true: true label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param y_pred: pred label image of shape (batch_size, z, y, x, num_class) or (batch_size, z, y, x, 1)
        :param volume_voxel:  volume of one voxel, default to 1.0
    :return: AVD, float
    """
    assert y_true.shape == y_pred.shape

    vol_g = np.sum(y_true, axis=y_true.shape[1:])
    vol_p = np.sum(y_pred, axis=y_pred.shape[1:])

    return np.mean(abs(vol_g - vol_p) / vol_g)
