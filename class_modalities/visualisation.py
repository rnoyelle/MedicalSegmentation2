import numpy as np
import matplotlib.pyplot as plt


def filter_img(img, vmax=None, vmin=None, cval_max=None, cval_min=None):
    if vmax:
        if cval_max:
            img[img > vmax] = cval_max
        else:
            img[img > vmax] = vmax

    if vmin:
        if cval_min:
            img[img < vmin] = cval_min
        else:
            img[img < vmin] = vmin

    return img


def make_grid(img):
    pass


def mip_pet(img, axis):
    mip = img.copy()
    mip = filter_img(mip, vmax=2.5)
    return np.max(mip, axis=axis)


def mip_ct(img, axis, remove_table=True):
    mip = img.copy()
    mip = filter_img(mip, vmax=1000.0, vmin=-1000.0)
    if axis == 1:
        if remove_table:
            ymin, ymax = int(0.25*img.shape[1]), int(0.75*img.shape[1])
            mip = mip[:, ymin:ymax, :]
    return np.max(mip, axis=axis)


def plot_scan(pet_img, ct_img, mask_img):

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Horizontally stacked subplots')

    axis = 0
    axes[0, 0].imshow(np.transpose(mip_pet(pet_img, axis=axis), (1, 0)))
    axes[0, 1].imshow(np.transpose(mip_ct(ct_img, axis=axis), (1, 0)))
    axes[0, 2].imshow(np.transpose(np.max(mask_img, axis=axis), (1, 0)))

    axis = 1
    axes[1, 0].imshow(np.transpose(mip_pet(pet_img, axis=axis), (1, 0)))
    axes[1, 1].imshow(np.transpose(mip_ct(ct_img, axis=axis), (1, 0)))
    axes[1, 2].imshow(np.transpose(np.max(mask_img, axis=axis), (1, 0)))

    axis = 2
    axes[2, 0].imshow(np.transpose(mip_pet(pet_img, axis=axis), (1, 0)))
    axes[2, 1].imshow(np.transpose(mip_ct(ct_img, axis=axis), (1, 0)))
    axes[2, 2].imshow(np.transpose(np.max(mask_img, axis=axis), (1, 0)))

    plt.show()


def show_scan(pet_array, ct_array):
    # (z, y, x)

    pet_img = pet_array.copy()
    ct_img = ct_array.copy()

    ct_img[ct_img > 1000.0] = 1000.0
    ct_img[ct_img < -1000.0] = -1000.0

    pet_img[pet_img > 10.0] = 10.0

    y_start, y_end = int(0.25 * ct_img.shape[1]), int(0.75 * ct_img.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(12, 12))

    axes[0, 0].imshow(np.max(pet_img, axis=1))
    axes[0, 0].set_title('Coronal PET')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.max(pet_img, axis=2))
    axes[0, 1].set_title('Sagittal PET')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(np.max(ct_img[:, y_start:y_end, :], axis=1), cmap='gray')
    axes[1, 0].set_title('Coronal CT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.max(ct_img, axis=2), cmap='gray')
    axes[1, 1].set_title('Sagittal CT')
    axes[1, 1].axis('off')

    axes[0, 2].imshow(np.max(pet_img, axis=0))
    axes[0, 2].set_title('Axial PET')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(np.max(ct_img, axis=0), cmap='gray')
    axes[1, 2].set_title('AxialCT')
    axes[1, 2].axis('off')

    # plt.tight_layout()
    plt.show()


def make_grid_mip(img):
    """
    Sagittal, Coronal, Axial MIP
    """
    pass


def sample_stack(img, rows=6, cols=6, start_with=10, show_every=10):
    fig, ax = plt.subplots(rows,cols,figsize=(12,12))
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(img[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def gif(img, mask_true, mask_pred):
    pass
