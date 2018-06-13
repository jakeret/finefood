import numpy as np

def read_glove_vecs(path):
    word_to_vec_map = {}
    word_to_index = {}
    with open(path, "r", encoding='utf-8') as fp:
        for i, line in enumerate(fp):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            word_to_index[word] = i
            word_to_vec_map[word] = coefs

    return word_to_index.keys(), word_to_index, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]  # number of training examples

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))

    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Loop over the words of sentence_words
        for j, w in enumerate(sentence_words):
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            j = j + 1
            if j >= max_len:
                break

    ### END CODE HERE ###

    return X_indices
