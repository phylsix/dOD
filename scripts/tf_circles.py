from dOD.tf_model.trainer import Trainer
from dOD.tf_model.datasets import circles
from dOD.tf_model.model import UNet

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


circle_model = UNet(input_shape=(200, 200, 1), kernel_shape=(3, 3),
                    root_feature=16, depth=3, drop_rate=0.2, num_classes=circles.CLASSES)
circle_model.build_net(True)
circle_model.compile(optimizer='rmsprop')


train_ds, validation_ds = circles.load_data(
    100, nx=200, ny=200, splits=(0.7, 0.3))

trainer = Trainer(save_ckpt=False)


trainer.fit(circle_model.net, train_ds,
            validation_dataset=validation_ds, epochs=5, batch_size=5)
