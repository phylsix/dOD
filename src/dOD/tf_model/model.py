from typing import Callable, Optional, Tuple, Union, List

import tensorflow as tf

from layers import (SequentialConv2DLayer,
                    Conv2DTransposeLayer,
                    CropConcatLayer)


class UNet:

    def __init__(self,
                 input_shape: Tuple(int, int, int),
                 kernel_shape: Tuple(int, int),
                 root_feature: int = 64,
                 nlayer: int = 2,
                 depth: int = 3,
                 drop_rate: float = 0.,
                 padding: str = "valid",
                 activation: Optional[Union[str, Callable]] = "relu",
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
        self.pool_size = pool_size
        self.num_classes = num_classes

        self.layers_downstream = {}
        # self.layers_upstream = {}

        self.net = None

    def buildNet(self, verbose: bool = False) -> None:
        inputs = tf.keras.Input(
            shape=self.ishape, name="inputs", dtype=tf.float32)

        x = inputs
        for idepth in range(self.depth):
            curr_feature = 2 ** idepth * self.root_feature
            conv_params = {
                "kernel_shape": (self.kshape[0], self.kshape[1], curr_feature),
                "nlayer": self.nlayer,
                "padding": self.padding,
                "strides": (self.pool_size, self.pool_size),
                "activation": self.activation,
                "drop_rate": self.drop_rate,
            }

            if idepth == 0:
                conv_params['input_shape'] = self.ishape

            x = SequentialConv2DLayer(**conv_params)(x)
            self.layers_downstream[idepth] = x

            if idepth < self.depth-1:
                x = tf.keras.layers.MaxPooling2D(
                    (self.pool_size, self.pool_size))(x)

            if verbose:
                print(f"-------- downstream layer {idepth} shape: {x.shape}")

        for idepth in range(self.depth - 1, -1, -1):
            curr_feature = 2 ** idepth * self.root_feature

            x = Conv2DTransposeLayer(kernel_shape=(self.pool_size, self.pool_size, curr_feature // 2),
                                     padding=self.padding,
                                     strides=self.pool_size,
                                     activation=self.activation)(x)

            x = CropConcatLayer()(self.layers_downstream[idepth], x)

            x = SequentialConv2DLayer(kernel_shape=(self.kshape[0], self.kshape[1], curr_feature),
                                      nlayer=self.nlayer,
                                      padding=self.padding,
                                      strides=(self.pool_size, self.pool_size),
                                      activation=self.activation,
                                      drop_rate=self.drop_rate)(x)

            if verbose:
                print(f"-------- upstream layer {idepth} shape: {x.shape}")

            # self.layers_upstream[self.depth-idepth-1] = x

        x = tf.keras.layers.Conv2D(filters=self.num_classes,
                                   kernel_size=(1, 1),
                                   padding=self.padding,
                                   activation=self.activation)(x)

        outputs = tf.keras.layers.Activation("softmax", name="outputs")(x)

        if verbose:
            print("-------- output shape:", outputs.shape)

        self.net = tf.keras.Model(inputs, outputs, name="UNet")

    def compile(self,
                loss: Optional[Union[str, Callable]] = "binary_crossentropy",
                optimizer: Optional[Union[str, Callable]] = None,
                metrics: Optional[List[Union[str, Callable]]] = None,
                **kwargs) -> None:

        if not self.net:
            self.buildNet()

        if optimizer is None:
            optimizer = tf.optimizers.Adam(**kwargs)

        if metrics is None:
            metrics = ['binary_crossentropy', 'binary_accuracy']

        self.net.compile(loss=loss,
                         optimizer=optimizer,
                         metrics=metrics)
