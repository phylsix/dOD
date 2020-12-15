from typing import Optional, Tuple, Union, List
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    Callback, ModelCheckpoint, History, TensorBoard)

from dOD.tf_model import layers
from dOD.tf_model.callbacks import TBLearningRate


def build_logpath(root: Optional[str] = 'unet') -> str:
    return os.path.join(root,
                        datetime.now().strftime("%y%m%d-%H%M%S"),
                        "{epoch:04d}.ckpt")


def get_output_shape(model: Model,
                     train_dataset: tf.data.Dataset) -> tf.Tensor:
    return model.predict(train_dataset.take(1).batch(batch_size=1)).shape


class Trainer:
    """Manager of training activities, mainly `fit`.

    Attributes:
        callbacks: a list of callbacks to call during training.
        logbase: a string of base path to save training logs.
        save_checkpoint: a boolean flag to ask `Trainer` to add a
            `ModelCheckpoint` callback by itself.
        opts: a dictionary to save addtional kwargs used for customization.
    """

    def __init__(self,
                 callbacks: Union[None, List[Callback]] = None,
                 logbase: Optional[str] = None,
                 save_checkpoint: bool = True,
                 **kwargs) -> None:

        self.callbacks = callbacks
        self.save_checkpoint = save_checkpoint
        self.opts = kwargs

        self.logpath = build_logpath(logbase) if logbase else None

    def build_callbacks(self) -> List[Callback]:
        """Build callbacks to be called for training.

        implictly includes:

        - `ModelCheckpoint` when `save_checkpoint` \
            (save_freq='epoch', save_best_only=True, save_weights_only=True)
        - `TensorBoard`
        - `TBLearningRate`

        if they are not included in the constructor.

        Returns:
            List[Callback]: assembled callbacks.
        """
        callbacks = self.callbacks if self.callbacks else []

        # ModelCheckpoint
        if self.save_checkpoint and self.logpath and \
                not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            callbacks.append(ModelCheckpoint(
                self.logpath,
                save_best_only=True,
                save_weights_only=True,
                save_freq=self.opts.get('save_freq', 'epoch'))
            )

        # TensorBoard
        if self.logpath:
            callbacks.append(TensorBoard(self.logpath))
            # callbacks.append(TBLearningRate(self.logpath))

        return callbacks

    def fit(self,
            model: Model,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            test_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            batch_size: int = 1,
            **kwargs) -> History:
        """Train the model with given setting, perform evaluation if
            `test_dataset` provided.

        Args:
            model: the model to be fit.
            train_dataset: a training dataset passed to `model.fit()`.
            validation_dataset:
                validation dataset passed to `model.fit()`. Defaults to None.
            test_dataset:
                test dataset passed to `model.evaluate()`. Defaults to None.
            epochs: number of epochs to train. Defaults to 10.
            batch_size: number of samples per batch. Defaults to 1.

        Returns:
            History: history of training.
        """

        out_shape = get_output_shape(model, train_dataset)[1:]

        train_dataset = train_dataset.map(
            layers.crop_labels_to_shape(out_shape)).batch(batch_size)
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        if validation_dataset:
            validation_dataset = validation_dataset.map(
                layers.crop_labels_to_shape(out_shape)).batch((batch_size))

        callbacks = self.build_callbacks()

        history = model.fit(train_dataset,
                            validation_data=validation_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            **kwargs)

        if test_dataset:
            test_dataset = test_dataset\
                .map(layers.crop_labels_to_shape(out_shape))\
                .batch(batch_size)
            model.evaluate(test_dataset)

        return history

    def evaluate(self,
                 model: Model,
                 test_dataset: Optional[tf.data.Dataset] = None,
                 shape: Tuple[int, int, int] = None) -> None:
        """evaluate test_dataset.

        Args:
            model: model to evaluate.
            test_dataset:
                test dataset passed to `model.evaluate()`. Defaults to None.
            shape: output shape. Defaults to None.
        """
        if test_dataset:
            if shape is None:
                shape = get_output_shape(model, test_dataset)[1:]
            test_dataset = test_dataset\
                .map(layers.crop_labels_to_shape(shape))\
                .batch(1)
            model.evaluate(test_dataset)
