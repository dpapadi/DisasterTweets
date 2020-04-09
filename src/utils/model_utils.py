import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.vocabulary.base import Vocabulary

torch.manual_seed(0)


glove_path = "glove/glove.twitter.27B.{}d.txt"


def get_train_test_sets(X, y, X_val, y_val, test_size, random_state):
    if X_val is not None and y_val is not None:
        return X, X_val, y, y_val
    return train_test_split(X, y, test_size=test_size, random_state=1)


# LSTM utils
def _get_glove(glove_dimensions):
    path = glove_path.format(glove_dimensions)
    glove_voc = []
    glove_weights = []
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            if len(values[1:]) == glove_dimensions:
                glove_voc.append(values[0])
                glove_weights.append(values[1:])
    print("GloVe embeddings imported.\nVector size:\t", len(glove_weights[-1]))
    return glove_voc, np.asarray(glove_weights, "float32")


def init_embeddings(emb_dimensions=200):
    vocab_list, weights = _get_glove(emb_dimensions)
    vocabulary = Vocabulary().add_list(vocab_list)
    emb_weights = nn.Parameter(torch.cat([torch.zeros(1, emb_dimensions), torch.tensor(weights)], dim=0))
    embeddings = nn.Embedding(vocabulary.size, emb_dimensions, padding_idx=0)
    embeddings.weight = emb_weights
    embeddings.weight.requires_grad = False
    return embeddings, vocabulary


# SNN utils
def get_use_embs(text, use_module):
    text = " ".join(text) if isinstance(text, list) else text
    text = [text.strip()]
    emb = use_module.signatures["response_encoder"](
        input=tf.constant(text),
        context=tf.constant(text))["outputs"].numpy()
    return emb


# Dataloader utils
def _sentence_to_index(l, sentence_length, vocabulary):
    lst = l if isinstance(l, list) else l.split()
    ret = [vocabulary.get_index(word) for word in lst]
    if len(ret) < sentence_length:
        ret += [0 for i in range(sentence_length - len(ret))]
    else:
        ret = ret[:sentence_length]
    return ret


def create_dataloader(X, y=None, shuffle=True, batch_size=64, vocabulary=None, sentence_length=None):
    if vocabulary:
        X = pd.Series(X)
        X = X.apply(lambda x: _sentence_to_index(x, sentence_length, vocabulary))
    X = np.vstack(X)
    dataset = (TensorDataset(torch.from_numpy(X), torch.from_numpy(y)) if isinstance(y, np.ndarray)
               else torch.from_numpy(X))
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
