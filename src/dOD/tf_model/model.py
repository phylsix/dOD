from typing import Callable, Optional, Tuple, Union, List

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU

from dOD.tf_model import layers


class UNet:
    """Model implementing U-net.

    Attributes:
        ishape: input sample shape.
        nx: input shape dimension 0.
        ny: input shape dimension 1.
        channels: input shape dimension 2.
        kshape: convolution kernel shape.
        root_feature: number of features of the first convolution after input.
        nlayer: number of convolutions in a sequence.
        depth: depth of U-net.
        drop_rate: drop rate after each convolution operation.
        padding: padding scheme for convolution, 'same' or 'valid'.
        activation: activation function to call for convolution.
        norm_type: normalization layer.
        pool_size: kernal size for MaxPooling2D and Conv2DTranspose.
        num_classes: number of possible classes for each pixel.
        net: model holder.
    """

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 kernel_shape: Tuple[int, int],
                 root_feature: int = 64,
                 nlayer: int = 2,
                 depth: int = 3,
                 drop_rate: float = 0.,
                 padding: str = "same",
                 activation: Optional[Union[str, Callable]] = LeakyReLU(0.2),
                 norm_type: Optional[Union[str, Callable]] = 'instancenorm',
                 pool_size: int = 2,
                 num_classes: int = 2) -> None:

        self.ishape = input_shape
        self.nx = input_shape[0]
        self.ny = input_shape[1]
        self.channels = input_shape[2]
        self.kshape = kernel_shape
        self.root_feature = root_feature
        self.nlayer = nlayer
        self.depth = depth
        self.drop_rate = drop_rate
        self.padding = padding
        self.activation = activation
        self.norm_type = norm_type
        self.pool_size = pool_size
        self.num_classes = num_classes

        self.net = None

    def build_net(self) -> None:

        downstream_layers = {}
        inputs = tf.keras.Input(shape=self.ishape, name="inputs")

        # downstream
        x = inputs
        for idepth in range(self.depth):
            curr_feature = 2 ** idepth * self.root_feature
            conv_params = {
                "kernel_shape": (self.kshape[0], self.kshape[1], curr_feature),
                "nlayer": self.nlayer,
                "padding": self.padding,
                "activation": self.activation,
                "drop_rate": self.drop_rate,
                "norm_type": self.norm_type,
            }

            if idepth == 0:
                conv_params['norm_type'] = None

            x = layers.SequentialConv2DLayer(**conv_params)(x)
            downstream_layers[idepth] = x

            if idepth < self.depth-1:
                x = tf.keras.layers.MaxPooling2D(
                    (self.pool_size, self.pool_size))(x)

        # upstream
        for idepth in range(self.depth - 2, -1, -1):
            curr_feature = 2 ** idepth * self.root_feature

            x = layers.Conv2DTransposeLayer(
                kernel_shape=(self.pool_size, self.pool_size, curr_feature),
                padding=self.padding,
                strides=self.pool_size,
                activation=self.activation,
                drop_rate=self.drop_rate,
                norm_type=self.norm_type)(x)

            x = layers.CropConcatLayer()(downstream_layers[idepth], x)

            x = layers.SequentialConv2DLayer(
                kernel_shape=(self.kshape[0], self.kshape[1], curr_feature),
                nlayer=self.nlayer,
                padding=self.padding,
                activation=self.activation,
                drop_rate=self.drop_rate,
                norm_type=self.norm_type)(x)

        x = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            strides=1,
            kernel_initializer=layers.get_kernel_initializer(
                self.root_feature, self.kshape),
            padding=self.padding)(x)

        x = tf.keras.layers.Activation(self.activation)(x)
        outputs = tf.keras.layers.Activation("softmax", name="outputs")(x)

        self.net = tf.keras.Model(inputs, outputs, name="UNet")

    def compile(self,
                loss: Optional[Union[str, Callable]] = "binary_crossentropy",
                optimizer: Optional[Union[str, Callable]] = None,
                metrics: Optional[List[Union[str, Callable]]] = None,
                **kwargs) -> None:

        if not self.net:
            self.build_net()

        if optimizer is None:
            optimizer = tf.optimizers.Adam(**kwargs)

        if metrics is None:
            metrics = ['binary_crossentropy', 'binary_accuracy']

        self.net.compile(loss=loss,
                         optimizer=optimizer,
                         metrics=metrics)

    def describle(self) -> None:
        """print the architecture of the net with output shape annotated.
        """

        if not self.net:
            self.build_net()

        N = len(self.net.layers)
        nSeqConv = 0
        for i, x in enumerate(self.net.layers):
            if i == 0:
                print(f'-------- input shape: {x.output_shape[0]}')
            if isinstance(x, layers.SequentialConv2DLayer):
                nSeqConv += 1
                if nSeqConv < self.depth:
                    name = 'downstream layer'
                    nlayer = nSeqConv
                elif nSeqConv == self.depth:
                    name = 'bottom layer'
                    nlayer = nSeqConv
                else:
                    name = 'upstream layer'
                    nlayer = 2 * self.depth - nSeqConv

                desc = [f'> layer {j} shape: {c.output_shape}'
                        for j, c in enumerate(x.layers)
                        if isinstance(c, tf.keras.layers.Conv2D)]
                desc.append(f'{name} {nlayer} shape: {x.output_shape}')

                for line in desc:
                    print(' ' * 4 * nlayer + line)
            if i == N - 1:
                print(f'-------- output shape: {x.output_shape}')
