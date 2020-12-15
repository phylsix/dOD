from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import TruncatedNormal


def get_kernel_initializer(filters: int,
                           kernel_size: Tuple[int, int]) -> TruncatedNormal:
    stddev = np.sqrt(2 / (kernel_size[0] * kernel_size[1] * filters))
    return TruncatedNormal(stddev=stddev)


class SequentialConv2DLayer(Layer):
    """Encapsulate a sequence of Conv2D operations.

    Attributes:
        ishape: input sample shape.
        nlayer: number of Conv2D layers.
        padding: padding scheme for Conv2D, 'same' or 'valid'.
        strides: strides arg for Conv2D.
        activation: activation arg for Conv2D.
        drop_rate: drop_rate arg for Conv2D.
        layers_conv2d: Conv2D layers.
        layers_dropout: Dropout layers.
        kernel_shape: convolution kernel shape.
        kernel_initializer: kernel_initializer arg for Conv2D.
    """

    def __init__(self,
                 kernel_shape: Tuple[int, int, int],
                 nlayer: int,
                 padding: str = 'same',
                 strides: int = 1,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 drop_rate: float = 0., **kwargs) -> None:
        super(SequentialConv2DLayer, self).__init__(**kwargs)

        self.ishape = None  # shape = [rows/height, cols/width, channels]
        self.nlayer = nlayer
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.drop_rate = drop_rate
        self.layers_conv2d = []
        self.layers_dropout = []
        self.kernel_shape = kernel_shape

        if "input_shape" in kwargs:
            self.ishape = kwargs["input_shape"]

        self.kernel_initializer = get_kernel_initializer(
            filters=kernel_shape[2], kernel_size=kernel_shape[:2])

        # First layer which contains input
        if self.ishape is None:
            self.layers_conv2d.append(
                tf.keras.layers.Conv2D(
                    filters=kernel_shape[2],
                    kernel_size=kernel_shape[:2],
                    strides=strides,
                    padding=padding,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer)
            )
        else:
            self.layers_conv2d.append(
                tf.keras.layers.Conv2D(
                    filters=kernel_shape[2],
                    kernel_size=kernel_shape[:2],
                    strides=strides,
                    padding=padding,
                    input_shape=self.ishape,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer)
            )

        if self.drop_rate > 0.:
            self.layers_dropout.append(
                tf.keras.layers.Dropout(self.drop_rate)
            )

        # The rest
        for _ in range(1, self.nlayer):
            self.layers_conv2d.append(
                tf.keras.layers.Conv2D(
                    filters=kernel_shape[2],
                    kernel_size=kernel_shape[:2],
                    strides=strides,
                    padding=padding,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer)
            )
            if self.drop_rate > 0.:
                self.layers_dropout.append(
                    tf.keras.layers.Dropout(self.drop_rate)
                )

    def call(self,
             inputs: tf.Tensor,
             training: bool = False,
             **kwargs) -> tf.Tensor:
        x = inputs

        for i in range(self.nlayer):
            x = self.layers_conv2d[i](x)
            if training and self.drop_rate > 0.:
                x = self.layers_dropout[i](x)

        return x

    def get_config(self) -> Dict[str, Any]:
        return dict(kernel_shape=self.kernel_shape,
                    nlayer=self.nlayer,
                    padding=self.padding,
                    strides=self.strides,
                    activation=self.activation,
                    drop_rate=self.drop_rate,
                    **super(SequentialConv2DLayer, self).get_config())


class Conv2DTransposeLayer(Layer):
    """Encapsulate Conv2DTranspose operation.

    Attributes:
        kernel_shape: kernel size and filters.
        padding: padding scheme for Conv2DTranspose.
        strides: strides arg for Conv2DTranspose.
        activation: activation arg for Conv2DTranspose.
        kernel_initializer: kernel_initializer arg for Conv2DTranspose.
        layer: holder of Conv2DTranspose.
    """

    def __init__(self,
                 kernel_shape: Tuple[int, int, int],
                 padding: str = 'valid',
                 strides: int = 2,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 **kwargs) -> None:
        super(Conv2DTransposeLayer, self).__init__(**kwargs)

        self.kernel_shape = kernel_shape
        self.padding = padding
        self.strides = strides
        self.activation = activation

        self.kernel_initializer = get_kernel_initializer(
            filters=kernel_shape[2], kernel_size=kernel_shape[:2])

        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=kernel_shape[2],
            kernel_size=kernel_shape[:2],
            strides=strides,
            padding=padding,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        x = inputs
        x = self.layer(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        return dict(kernel_shape=self.kernel_shape,
                    padding=self.padding,
                    strides=self.strides,
                    activation=self.activation,
                    **super(Conv2DTransposeLayer, self).get_config())


class CropConcatLayer(Layer):
    """Crop a downstream output tensor and concat with a upstream output.
    """

    def call(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """implement crop-concat operation.

        Args:
            x1: downstream output.
            x2: upstream output.

        Returns: tf.Tensor
        """
        # the shape of the tensor is [batch, rows, cols, channels]
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)

        h_diff = x1_shape[1] - x2_shape[1]
        w_diff = x1_shape[2] - x2_shape[2]

        h_diff_half = h_diff // 2
        w_diff_half = w_diff // 2

        down_layer_cropped = x1[:,
                                h_diff_half: (x1_shape[1] - h_diff_half),
                                w_diff_half: (x1_shape[2] - w_diff_half),
                                :]

        return tf.concat([down_layer_cropped, x2], axis=-1)


def crop_to_shape(X: tf.Tensor, shape: Tuple[int, int, int]) -> tf.Tensor:

    x_diff = X.shape[0] - shape[0]
    y_diff = X.shape[1] - shape[1]

    if x_diff == y_diff == 0:
        return X

    x_diff_half = x_diff // 2
    y_diff_half = y_diff // 2

    return X[x_diff_half:x_diff_half - x_diff,
             y_diff_half:y_diff_half - y_diff]


def crop_labels_to_shape(shape: Tuple[int, int, int]) -> Callable:
    def crop(img, label):
        return img, crop_to_shape(label, shape)
    return crop


def crop_image_labels_to_shape(shape: Tuple[int, int, int]) -> Callable:
    def crop(img, label):
        return crop_to_shape(img, shape), crop_to_shape(label, shape)
    return crop
