import numpy as np

def read_glove_vecs(path):
    word_to_vec_map = {}
    word_to_index = {}
    with open(path, "r") as fp:
        for i, line in enumerate(fp):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            word_to_index[word] = i
            word_to_vec_map[word] = coefs

    return word_to_index.keys(), word_to_index, word_to_vec_map
