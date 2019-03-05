import random

import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LinearRegression


def fit_w2v_regression(model1, model2, sample_size):
    """Given two gensim Word2Vec models, fit a regression model using a subset of the vocabulary.
    The size of this subset is given by the samplesize parameter, which can specify either a
    percentage of the common vocab to use, or the number of words to use.
    ::param model1:: a gensim `KeyedVectors` instance for the LHS
    ::param model2:: a gensim `KeyedVectors` instance for the RHS
    ::param sample_size:: a float or int specifying how much of the vocab to use to fit the model.
    ::returns:: a `sklearn.linear_model.LinearRegression` object.
    """
    common_vocab = set(model1.vocab.keys()).intersection(set(model2.vocab.keys()))
    if "</s>" in common_vocab:
        common_vocab.remove("</s>")
    if type(sample_size) == float:
        sample_size = int(sample_size * len(common_vocab))
    try:
        d1 = model1.vector_size
    except AttributeError:
        d1 = None
    if d1 is None:
        d1 = model1.syn0.shape[1]
    try:
        d2 = model2.vector_size
    except AttributeError:
        d2 = None
    if d2 is None:
        d2 = model2.syn0.shape[1]
    sample = random.sample(common_vocab, sample_size)
    X = np.ndarray((sample_size, d1), dtype=np.float32)
    Y = np.ndarray((sample_size, d2), dtype=np.float32)
    for i, word in enumerate(sample):
        X[i, :] = model1[word]
        Y[i, :] = model2[word]
    regression = LinearRegression()
    regression.fit(X, Y)
    return regression


def apply_w2v_regression(model, regression):
    """Given a word2vec model and a linear regression, apply that regression to all the vectors
    in the model.
    ::param model:: A gensim `KeyedVectors` or `Word2Vec` instance
    ::param regression:: A `sklearn.linear_model.LinearRegression` instance
    ::returns:: A gensim `KeyedVectors` instance
    """
    aligned_model = KeyedVectors()  # Word2Vec()
    aligned_model.vocab = model.vocab.copy()
    aligned_model.vector_size = model.vector_size
    aligned_model.index2word = model.index2word
    # aligned_model.reset_weights()
    aligned_model.syn0 = regression.predict(model.syn0).astype(np.float32)
    return aligned_model


def align_embeddings(base_embed, other_embed, sample_size=1):
    """Fit the regression that aligns model1 and model2."""
    regression = fit_w2v_regression(base_embed, other_embed, sample_size)
    aligned_model = apply_w2v_regression(base_embed, regression)
    return aligned_model
