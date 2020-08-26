

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


def mip(img):
    pass

def make_grid_mip(img):
    """
    Sagittal, Coronal, Axial MIP
    """
    pass

def gif(img, mask_true, mask_pred):
    pass
