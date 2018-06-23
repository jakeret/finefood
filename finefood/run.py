import argparse
import os

import h5py
import tensorflow as tf

import numpy as np
import pandas as pd

import utils
import score_model

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from polyaxon_helper import send_metrics, get_log_level
from polyaxon_helper import get_data_path, get_outputs_path


def set_logging(log_level=None):
    if log_level == 'INFO':
        log_level = tf.logging.INFO
    elif log_level == 'DEBUG':
        log_level = tf.logging.DEBUG
    elif log_level == 'WARN':
        log_level = tf.logging.WARN
    else:
        log_level = 'INFO'

    tf.logging.set_verbosity(log_level)


set_logging(get_log_level())


def train_model(model, model_type, X_train, X_test, y_train, y_test, **kwargs):
    checkpoint = get_checkpoint_callback(model_type)

    tensorboard = get_tensorboard_callback(model_type, kwargs)


    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint, tensorboard, early_stopping],
                        **kwargs)

    return history


def get_tensorboard_callback(model_type, kwargs):
    outputs_path = get_outputs_path()
    if outputs_path is None:
        outputs_path = os.path.join("./output", model_type)
    print("Storing tensorflow logs to", outputs_path)

    tensorboard = TensorBoard(log_dir=outputs_path,
                              histogram_freq=0,
                              batch_size=kwargs.get("batch_size", 32),
                              write_graph=True)
    return tensorboard


def get_checkpoint_callback(model_type):
    filename = "model-%s-{epoch:02d}-{val_acc:.2f}.hdf5"%model_type
    outputs_path = get_outputs_path()
    if outputs_path is None:
        outputs_path = "./output"
    filepath = os.path.join(outputs_path, filename)
    print("Storing checkpoints to", filepath)

    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    return checkpoint


def launch(model_type, epochs, batch_size, learning_rate, dropout, trainable_embbedings=True, sample_size=None, max_len=1000):
    print("Starting with: model-type %s, epochs %s, batch_size %s, learning_rate %s, dropout %s"%(model_type, epochs, batch_size, learning_rate, dropout))

    model = score_model.build_model(model_type, max_len, dropout, trainable_embbedings)
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=["acc", "mae"])
    model.summary()

    X_train, X_test, y_train, y_test = load_data(sample_size, max_len)
    print("Data and label shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    history = train_model(model, model_type, X_train, X_test, y_train, y_test,
                          epochs=epochs, batch_size=batch_size, shuffle=True)

    try:
        send_metrics(loss=history.history["loss"][-1],
                     val_loss=history.history["val_loss"][-1],
                     accuracy=history.history["acc"][-1],
                     val_accuracy=history.history["val_acc"][-1])
    except Exception:
        pass


def load_data(sample_size, max_len, load_from_csv=False):
    data_path = get_data_path()
    if data_path is None:
        data_path = "./data"

    path = os.path.join(data_path, "Reviews.h5")
    print("Loading data from", path)
    with h5py.File(path, "r") as fp:
        text_indices = fp["text_indices"].value
        scores_oh = fp["scores_oh"].value

        idx = np.arange(text_indices.shape[0])
        np.random.shuffle(idx)
        text_indices = text_indices[idx]
        scores_oh = scores_oh[idx]

    if sample_size is not None:
        text_indices = text_indices[:int(sample_size), :max_len]
        scores_oh = scores_oh[:int(sample_size)]

    X_train, X_test, y_train, y_test = train_test_split(text_indices, scores_oh, test_size=.2)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_type',
        default="cnn_lstm",
        choices=["lstm", "2layer_lstm", "cnn_lstm", "cnn", "1layer_cnn"]
    )
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--dropout',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--trainable_embbedings',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--num_epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--max_len',
        default=1000,
        type=int
    )
    parser.add_argument('--sample_size')

    args = parser.parse_args()

    launch(args.model_type,
           args.num_epochs,
           args.batch_size,
           args.learning_rate,
           args.dropout,
           args.trainable_embbedings,
           args.sample_size,
           args.max_len
           )
