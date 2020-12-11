from typing import Optional, Union, List
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, History

from .layers import crop_labels_to_shape


def build_logpath(root: Optional[str] = 'unet') -> str:
    return os.path.join(root, datetime.now().strftime("%y%m%d-%H%M%S"))

class Trainer:
    def __init__(self,
                 callbacks: Union[None, List[Callback]] = None,
                 logbase : Optional[str] = None,
                 save_ckpt: bool = True,
                 save_TB_learningrate: bool = True,
                 save_TB_imagesummary: bool = True) -> None:

        self.callbacks = callbacks
        self.save_ckpt = save_ckpt

        # TODO implement those callbacks
        self.save_TB_learningrate = save_TB_learningrate
        self.save_TB_imagesummary = save_TB_imagesummary

        self.logpath = build_logpath(logbase)

    def get_output_shape(self,
                         model: Model,
                         train_dataset: tf.data.Dataset) -> tf.Tensor:
        return model.predict(train_dataset.take(1).batch(batch_size=1)).shape

    def build_callbacks(self,
                        train_dataset: tf.data.Dataset,
                        validation_dataset: Optional[tf.data.Dataset]) -> List[Callback]:
        callbacks = self.callbacks if self.callbacks else []
        if self.save_ckpt:
            callbacks.append(ModelCheckpoint(self.logpath,
                                             save_best_only=True))

        if self.save_TB_learningrate:
            pass
            # callbacks.append()

        if self.save_TB_imagesummary:
            pass
            # callbacks.append()
            if validation_dataset:
                pass
                # callbacks.append()

        return callbacks



    def fit(self,
            model: Model,
            train_dataset: tf.data.Dataset,
            validation_dataset: Optional[tf.data.Dataset] = None,
            test_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            batch_size: int = 1,
            **kwargs) -> History:

        out_shape = self.get_output_shape(model, train_dataset)

        train_dataset = train_dataset.map(
            crop_labels_to_shape(out_shape)).batch(batch_size)
        if validation_dataset:
            validation_dataset = validation_dataset.map(
                crop_labels_to_shape(out_shape)).batch((batch_size))

        callbacks = self.build_callbacks(train_dataset, validation_dataset)

        history = model.fit(train_dataset,
                            validation_dataset=validation_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            **kwargs)

        if test_dataset:
            test_dataset = test_dataset\
                .map(crop_labels_to_shape(out_shape))\
                .batch(batch_size)
            model.evaluate(test_dataset)

        return history
