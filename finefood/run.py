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
from keras.callbacks import ModelCheckpoint, TensorBoard

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


def train_model(model, X_train, X_test, y_train, y_test, **kwargs):
    checkpoint = get_checkpoint_callback()

    tensorboard = get_tensorboard_callback(kwargs)

    callbacks = [checkpoint, tensorboard]

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        **kwargs)

    return history


def get_tensorboard_callback(kwargs):
    outputs_path = get_outputs_path()
    if outputs_path is None:
        outputs_path = "./output"
    print("Storing tensorflow logs to", outputs_path)

    tensorboard = TensorBoard(log_dir=outputs_path,
                              histogram_freq=0,
                              batch_size=kwargs.get("batch_size", 32),
                              write_graph=True)
    return tensorboard


def get_checkpoint_callback():
    filename = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
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


def launch(epochs, batch_size, learning_rate, dropout, sample_size=None):
    print("Starting with: epochs %s, batch_size %s, learning_rate %s, dropout %s"%(epochs, batch_size, learning_rate, dropout))

    data_path = get_data_path()
    if data_path is None:
        data_path = "./data"
    path = os.path.join(data_path, "glove.6B.50d.txt")
    print("Loading glove from", path)

    corpus, word_to_index, word_to_vec_map = utils.read_glove_vecs(path)

    X_train, X_test, y_train, y_test = load_data(corpus, word_to_index, sample_size)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    max_len = X_train.shape[1]

    model = score_model.build_model((max_len,), dropout, word_to_vec_map, word_to_index)
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["acc", "mae"])
    model.summary()

    history = train_model(model, X_train, X_test, y_train, y_test,
                          epochs=epochs, batch_size=batch_size, shuffle=True)

    try:
        send_metrics(loss=history.history["loss"][-1],
                     val_loss=history.history["val_loss"][-1],
                     accuracy=history.history["acc"][-1],
                     val_accuracy=history.history["val_acc"][-1])
    except Exception:
        pass


def load_data(corpus, word_to_index, sample_size, load_from_csv=False, max_len=250):
    data_path = get_data_path()
    if data_path is None:
        data_path = "./data"

    if load_from_csv:
        import preprocessing
        path = os.path.join(data_path, "Reviews.csv")

        print("Loading data from", path)

        df = pd.read_csv(path).set_index("Id")
        if sample_size is not None:
            df = df.sample(n=sample_size)
        else:
            sample_size = df.size

        print("Sample size", sample_size)

        clean_texts = np.array([preprocessing.clean_text(t, corpus) for t in df.Text])
        scores = df.Score.values
        scores_oh = np_utils.to_categorical(scores)[:, 1:]
        text_indices = utils.sentences_to_indices(clean_texts, word_to_index, max_len)
    else:
        path = os.path.join(data_path, "Reviews.h5")
        print("Loading data from", path)
        with h5py.File(path, "r") as fp:
            text_indices = fp["text_indices"].value
            scores_oh = fp["scores_oh"].value

        if sample_size is not None:
            text_indices = text_indices[:int(sample_size)]
            scores_oh = scores_oh[:int(sample_size)]

    X_train, X_test, y_train, y_test = train_test_split(text_indices, scores_oh, test_size=.2)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
        '--num_epochs',
        default=10,
        type=int
    )
    parser.add_argument('--sample_size')

    args = parser.parse_args()

    launch(args.num_epochs,
           args.batch_size,
           args.learning_rate,
           args.dropout,
           args.sample_size,
           )
