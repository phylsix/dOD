from typing import Tuple
import functools

import tensorflow as tf
import tensorflow_datasets as tfds

# https://www.tensorflow.org/tutorials/images/segmentation

# tfds.core.DatasetInfo(
#     name='oxford_iiit_pet',
#     version=3.2.0,
#     description='The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200
# images for each class. The images have large variations in scale, pose and
# lighting. All images have an associated ground truth annotation of breed.',
#     homepage='http://www.robots.ox.ac.uk/~vgg/data/pets/',
#     features=FeaturesDict({
#         'file_name': Text(shape=(), dtype=tf.string),
#         'image': Image(shape=(None, None, 3), dtype=tf.uint8),
#         'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=37),
#         'segmentation_mask': Image(shape=(None, None, 1), dtype=tf.uint8),
#         'species': ClassLabel(shape=(), dtype=tf.int64, num_classes=2),
#     }),
#     total_num_examples=7349,
#     splits={
#         'test': 3669,
#         'train': 3680,
#     },
#     supervised_keys=('image', 'label'),
#     citation="""@InProceedings{parkhi12a,
#                                author = "Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C.~V.",
#                                title = "Cats and Dogs",
#                                booktitle = "IEEE Conference on Computer Vision and Pattern Recognition",
#                                year = "2012",
#                                }""",
#     redistribution_info=,
# )


tfds.disable_progress_bar()

IMAGE_SIZE = (128, 128)
CHANNELS = 3
CLASSES = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _normalize(img, mask):
    img = tf.cast(img, tf.float32) / 255.
    mask -= 1
    return img, mask


def _load_image(data, training=True):
    img = tf.image.resize(data['image'], IMAGE_SIZE)
    mask = tf.image.resize(data['segmentation_mask'], IMAGE_SIZE)

    if training and tf.random.uniform(()) > 0.5:
        # random mirroring
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    return _normalize(img, mask)


def load_data(
    buffer_size: int = 1000) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    ds, ds_info = tfds.load('oxford_iiit_pet', with_info=True)
    train_ds = ds['train'].map(_load_image, num_parallel_calls=AUTOTUNE)
    test_ds = ds['test'].map(functools.partial(_load_image, training=False))

    train_ds = train_ds.cache().shuffle(buffer_size).take(
        ds_info.splits['train'].num_examples)
    return train_ds, test_ds
