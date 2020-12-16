from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import TruncatedNormal


def get_kernel_initializer(filters: int,
                           kernel_size: Tuple[int, int]) -> TruncatedNormal:
    stddev = np.sqrt(2 / (kernel_size[0] * kernel_size[1] * filters))
    return TruncatedNormal(stddev=stddev)


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)

    Attributes:
        epsilon: small number
        scale:
        offset:
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape) -> None:
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True
        )

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class SequentialConv2DLayer(Layer):
    """Encapsulate a sequence of Conv2D operations.

    (Conv2D => Normalization => Dropout => Activation) * nlayer
    Default:     batchNorm       0.5        relu           2


    Attributes:
        ishape: input sample shape.
        nlayer: number of Conv2D layers.
        padding: padding scheme for Conv2D, 'same' or 'valid'.
        strides: strides arg for Conv2D.
        activation: activation function.
        drop_rate: drop_rate arg for Conv2D.
        norm_type: normalization layer, 'batchnorm' or 'instancenorm'.
        sequence: list of layers.
        kernel_shape: convolution kernel shape.
        kernel_initializer: kernel_initializer arg for Conv2D.
    """

    def __init__(self,
                 kernel_shape: Tuple[int, int, int],
                 nlayer: int = 2,
                 padding: str = 'same',
                 strides: int = 1,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 drop_rate: float = 0.5,
                 norm_type: Optional[Union[str, Callable]] = 'batchnorm',
                 **kwargs) -> None:
        super(SequentialConv2DLayer, self).__init__(**kwargs)

        self.ishape = None  # shape = [rows/height, cols/width, channels]
        self.nlayer = nlayer
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.drop_rate = drop_rate
        self.norm_type = norm_type
        self.sequence = tf.keras.Sequential()
        self.kernel_shape = kernel_shape

        self.kernel_initializer = get_kernel_initializer(
            filters=kernel_shape[2], kernel_size=kernel_shape[:2])

        # if self.ishape is not None:
        #     self.sequence.add(tf.keras.layers.Input(shape=self.ishape))

        for _ in range(self.nlayer):
            self.sequence.add(
                tf.keras.layers.Conv2D(
                    filters=kernel_shape[2],
                    kernel_size=kernel_shape[:2],
                    strides=strides,
                    padding=padding,
                    kernel_initializer=self.kernel_initializer)
            )

            if self.norm_type:
                if isinstance(self.norm_type, str):
                    if self.norm_type.lower() == 'batchnorm':
                        self.sequence.add(tf.keras.layers.BatchNormalization())
                    if self.norm_type.lower() == 'instancenorm':
                        self.sequence.add(InstanceNormalization())
                elif isinstance(self.norm_type, Callable):
                    self.sequence.add(self.norm_type)

            if self.drop_rate > 0.:
                self.sequence.add(tf.keras.layers.Dropout(self.drop_rate))

            if self.activation:
                if isinstance(self.activation, str):
                    self.sequence.add(
                        tf.keras.layers.Activation(self.activation))
                elif isinstance(self.activation, Callable):
                    self.sequence.add(self.activation)

    @property
    def layers(self):
        return self.sequence.layers

    def call(self,
             inputs: tf.Tensor,
             training: bool = False,
             **kwargs) -> tf.Tensor:
        x = self.sequence(inputs)
        return x

    def get_config(self) -> Dict[str, Any]:
        return dict(kernel_shape=self.kernel_shape,
                    nlayer=self.nlayer,
                    padding=self.padding,
                    strides=self.strides,
                    activation=self.activation,
                    drop_rate=self.drop_rate,
                    norm_type=self.norm_type,
                    **super(SequentialConv2DLayer, self).get_config())


class Conv2DTransposeLayer(Layer):
    """Encapsulate Conv2DTranspose operation.

    Conv2DTranspose => Normalization => Dropout => Activation
    Default:              batchNorm        0.5        relu

    Attributes:
        kernel_shape: kernel size and filters.
        padding: padding scheme for Conv2DTranspose.
        strides: strides arg for Conv2DTranspose.
        activation: activation arg for Conv2DTranspose.
        drop_rate: dropout rate.
        norm_type: normalization layer, 'batchnorm' or 'instancenorm'.
        kernel_initializer: kernel_initializer arg for Conv2DTranspose.
        sequence: list of layers.
    """

    def __init__(self,
                 kernel_shape: Tuple[int, int, int],
                 padding: str = 'valid',
                 strides: int = 2,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 drop_rate: float = 0.5,
                 norm_type: Optional[Union[str, Callable]] = 'batchnorm',
                 **kwargs) -> None:
        super(Conv2DTransposeLayer, self).__init__(**kwargs)

        self.kernel_shape = kernel_shape
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.drop_rate = drop_rate
        self.norm_type = norm_type
        self.sequence = tf.keras.Sequential()

        self.kernel_initializer = get_kernel_initializer(
            filters=kernel_shape[2], kernel_size=kernel_shape[:2])

        self.sequence.add(tf.keras.layers.Conv2DTranspose(
            filters=kernel_shape[2],
            kernel_size=kernel_shape[:2],
            strides=strides,
            padding=padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False)
        )

        if self.norm_type:
            if isinstance(self.norm_type, str):
                if self.norm_type.lower() == 'batchnorm':
                    self.sequence.add(tf.keras.layers.BatchNormalization())
                if self.norm_type.lower() == 'instancenorm':
                    self.sequence.add(InstanceNormalization())
            elif isinstance(self.norm_type, Callable):
                self.sequence.add(self.norm_type)

        if self.drop_rate > 0.:
            self.sequence.add(tf.keras.layers.Dropout(self.drop_rate))

        if self.activation:
            if isinstance(self.activation, str):
                self.sequence.add(
                    tf.keras.layers.Activation(self.activation))
            elif isinstance(self.activation, Callable):
                self.sequence.add(self.activation)

    @property
    def layers(self):
        return self.sequence.layers

    def call(self,
             inputs: tf.Tensor,
             training: bool = False,
             **kwargs) -> tf.Tensor:
        x = self.sequence(inputs)
        return x

    def get_config(self) -> Dict[str, Any]:
        return dict(kernel_shape=self.kernel_shape,
                    padding=self.padding,
                    strides=self.strides,
                    activation=self.activation,
                    drop_rate=self.drop_rate,
                    norm_type=self.norm_type,
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
