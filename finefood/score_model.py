import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv1D, MaxPooling1D
from keras.layers import Embedding


def pretrained_embedding_layer(word_to_vec_map, word_to_index, max_len):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len,
                                emb_dim,
                                weights=[emb_matrix],
                                input_length=max_len,
                                trainable=False)

    # # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    # embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    # embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def build_2layer_lstm_model(input_shape, dropout, word_to_vec_map, word_to_index):
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
    embedding_layer =  pretrained_embedding_layer(word_to_vec_map, word_to_index, input_shape[0])

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
    X =  Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence_indices, X)

    return model

def build_cnn_lstm_model(input_shape, dropout, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype=np.int32)

    embedding_layer =  pretrained_embedding_layer(word_to_vec_map, word_to_index, input_shape[0])

    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    X = Dropout(dropout)(embeddings)
    X = Conv1D(64, 5, activation='relu')(X)
    X = MaxPooling1D(pool_size=4)(X)
    X = LSTM(100)(X)
    X = Dense(5, activation='sigmoid')(X)
    X = Activation('softmax')(X)
    model = Model(sentence_indices, X)

    return model
