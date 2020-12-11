from typing import List, Tuple

import numpy as np
import tensorflow as tf

CHANNELS = 1
CLASSES = 2


def _create_img_mask(nx: int, ny: int,
                     ncircles: int = 10,
                     radius_range: Tuple[int, int] = (3, 10),
                     border: int = 32,
                     sigma: int = 20) -> Tuple[np.array, np.array]:
    img = np.ones((nx, ny, 1))
    mask = np.zeros((nx, ny), dtype=np.bool)

    for _ in ncircles:
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(*radius_range)
        h = np.random.randint(1, 255)

        y, x = np.ogrid[-a:nx - a, -b: ny - b]
        m = x ** 2 + y ** 2 <= r**2
        mask = np.logical_or(mask, m)

        img[m] = h

    img += np.random.normal(scale=sigma, size=img.shape)
    img -= np.amin(img)
    img /= np.amax(img)

    return img, mask


def _create_samples(N: int, nx: int, ny: int,
                    **kwargs) -> Tuple[np.array, np.array]:
    imgs = np.empty((N, nx, ny, 1))
    labels = np.empty((N, nx, ny, 2))
    for i in range(N):
        img, mask = _create_img_mask(nx, ny, **kwargs)
        imgs[i] = img
        labels[i, ..., 0] = ~mask
        labels[i, ..., 1] = mask
    return imgs, labels


def load_data(N: int,
              splits: Tuple[float] = (0.7, 0.2, 0.1),
              **kwargs) -> List[tf.data.Dataset]:
    return [
        tf.data.Dataset.from_tensor_slices(
            _create_samples(int(N * split), **kwargs)
        ) for split in splits
    ]
