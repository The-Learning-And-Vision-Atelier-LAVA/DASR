import random
import numpy as np
import skimage.color as sc
import torch


def get_patch(img, patch_size=48, scale=1):
    th, tw = img.shape[:2]  ## HR image

    tp = round(scale * patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :]


def set_channel(img, n_channels=3):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    c = img.shape[2]
    if n_channels == 1 and c == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channels == 3 and c == 1:
        img = np.concatenate([img] * n_channels, 2)

    return img


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)

    return tensor


def augment(img, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip: img = img[:, ::-1, :]
    if vflip: img = img[::-1, :, :]
    if rot90: img = img.transpose(1, 0, 2)

    return img

