import os

import h5py
import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv1D, MaxPooling1D, Flatten
from keras.layers import Embedding
from polyaxon_helper import get_data_path


def build_model(model_type, max_len, dropout):
    if model_type == "2layer_lstm":
        model = build_2layer_lstm_model((max_len,), dropout)
    elif model_type == "cnn_lstm":
        model = build_cnn_lstm_model((max_len,), dropout)
    elif model_type == "cnn":
        model = build_cnn_model((max_len,), dropout)
    else:
        raise "Unknown model: %s"%model_type

    return model

def pretrained_embedding_layer(max_len):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    data_path = get_data_path()
    if data_path is None:
        data_path = "./data"
    path = os.path.join(data_path, "Reviews.h5")
    print("Loading glove from", path)

    with h5py.File(path, "r") as fp:
        embedding_matrix = fp["embedding_matrix"].value

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=max_len,
                                trainable=False)

    return embedding_layer

def build_2layer_lstm_model(input_shape, dropout):
    """
    Function creating the model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer =  pretrained_embedding_layer(input_shape[0])

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(dropout)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    X = Dropout(dropout)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5, activation='softmax')(X)
    # Add a softmax activation
    # X =  Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)

    return model

def build_cnn_lstm_model(input_shape, dropout):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    embedding_layer =  pretrained_embedding_layer(input_shape[0])

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    X = Dropout(dropout)(embeddings)
    X = Conv1D(128, 5, activation='relu')(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = LSTM(128)(X)
    X = Dense(5, activation='softmax')(X)
    # X = Activation('softmax')(X)
    model = Model(sentence_indices, X)

    return model

def build_cnn_model(input_shape, dropout):
    sequence_input = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(input_shape[0])
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(5, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model
