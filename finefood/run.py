import argparse
import os

import tensorflow as tf

import numpy as np
import pandas as pd

from finefood import utils
from finefood import score_model
from finefood import preprocessing

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
    tensorboard = TensorBoard(log_dir=get_outputs_path(),
                              histogram_freq=0,
                              batch_size=kwargs.get("batch_size", 32),
                              write_graph=True)
    return tensorboard


def get_checkpoint_callback():
    filename = "model-{epoch:02d}-{val_acc:.2f}.hdf5"
    filepath = os.path.join(get_data_path(), filename)
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    return checkpoint


def launch(epochs, batch_size, learning_rate, dropout):
    max_len = 10

    corpus, word_to_index, word_to_vec_map = utils.read_glove_vecs("./data/glove.6B/glove.6B.50d.txt")

    X_test, X_train, y_test, y_train = load_data(corpus, max_len, word_to_index)

    model = score_model.build_model((max_len,), dropout, word_to_vec_map, word_to_index)
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    history = train_model(model, X_train, X_test, y_train, y_test,
                          epochs=epochs, batch_size=batch_size, shuffle=True)

    send_metrics(loss=history.history["loss"][-1],
                 val_loss=history.history["val_loss"][-1],
                 accuracy=history.history["acc"][-1],
                 val_accuracy=history.history["val_acc"][-1])


def load_data(corpus, max_len, word_to_index):
    df = pd.read_csv("./data/Reviews.csv").set_index("Id")
    df = df.sample(n=1000)
    clean_texts = np.array([preprocessing.clean_text(t, corpus) for t in df.Text])
    scores = df.Score.values
    scores_oh = np_utils.to_categorical(scores)[:, 1:]
    text_indices = preprocessing.sentences_to_indices(clean_texts, word_to_index, max_len)
    X_train, X_test, y_train, y_test = train_test_split(text_indices, scores_oh, test_size=.2)
    return X_test, X_train, y_test, y_train


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

    args = parser.parse_args()
    arguments = args.__dict__

    batch_size = arguments.pop('batch_size')
    learning_rate = arguments.pop('learning_rate')
    dropout = arguments.pop('dropout')
    num_epochs = arguments.pop('num_epochs')

    launch(num_epochs, batch_size, learning_rate, dropout)