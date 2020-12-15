from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

class TBLearningRate(TensorBoard):
    """callbacks to record the learning rate at the end of epoch.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['learning_rate'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs=logs)