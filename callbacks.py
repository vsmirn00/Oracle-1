import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def f1_score(y_true, y_pred):
    y_pred = tf.round(tf.keras.backend.clip(y_pred, 0, 1))
    tp = tf.reduce_sum(tf.cast(y_true*y_pred, tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1-y_true)*y_pred, tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true*(1-y_pred), tf.float32), axis=0)
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    f1_macro = tf.reduce_mean(f1)

    return f1_macro

class CustomCallbacks(Callback):
    def __init__(self, monitor='accuracy', mode='max', patience=5, verbose=1, factor=0.1, min_lr=0.00001):
        super(CustomCallbacks, self).__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.verbose = verbose
        self.factor = factor
        self.min_lr = min_lr
        self.best = None  # add this line
        self.early_stop = EarlyStopping(monitor=self.monitor, mode=self.mode, patience=self.patience, verbose=self.verbose)
        self.lr_reduce = ReduceLROnPlateau(monitor=self.monitor, mode=self.mode, factor=self.factor, patience=self.patience, verbose=self.verbose, min_lr=self.min_lr)
        self.model_checkpoint = ModelCheckpoint('best_model.h5', monitor=self.monitor, mode=self.mode, save_best_only=True, verbose=self.verbose)
        self.tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4 and logs.get(self.monitor) > 0.99:
            print('\nLoss is low and ' + self.monitor + ' is high, so cancelling training!')
            self.model.stop_training = True

        if self.best is None:
            self.best = logs.get(self.monitor)
        elif self.mode == 'max' and logs.get(self.monitor) > self.best:
            self.best = logs.get(self.monitor)
        elif self.mode == 'min' and logs.get(self.monitor) < self.best:
            self.best = logs.get(self.monitor)

        self.early_stop.best = self.best  # add this line
        self.early_stop.on_epoch_end(epoch, logs=logs)
        self.lr_reduce.on_epoch_end(epoch, logs=logs)
        self.model_checkpoint.on_epoch_end(epoch, logs=logs)
        self.tensorboard_callback.on_epoch_end(epoch, logs=logs)
    
    def on_train_end(self, logs={}):
        self.tensorboard_callback.on_train_end(logs=logs)