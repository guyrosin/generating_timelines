import os
from gensim.models import KeyedVectors, Word2Vec

from word2vec_model import Word2VecModelBase


class Word2VecSpecificModel(Word2VecModelBase):
    def __init__(self, year=None, dir_path=None, file=None, model=None, wv=True):
        if model:
            self.model = model
            self.wv = model.wv
            return
        # load the model from the given path
        if not file and year:
            file = 'word2vec-nyt-%i.model' % year
        self.MODEL_PATH = os.path.join(dir_path, file)

        if os.path.exists(self.MODEL_PATH):  # load the model
            if wv:
                self.wv = KeyedVectors.load(self.MODEL_PATH).wv
            else:
                self.model = Word2Vec.load(self.MODEL_PATH)
            # re-save the model with precomputed L2-normalized vectors
            # self.model.init_sims(replace=True)
            # self.model.save(os.path.join(dir_path, 'new', file))
        else:
            print("model doesn't exist!")

    def get_word_vector(self, word):
        try:
            if self.wv:
                return self.wv[word]
            else:
                return self.model.wv[word]
        except KeyError:
            return None

    def contains(self, word):
        if self.wv:
            return word in self.wv
        else:
            return word in self.model.wv

    def contains_all_words(self, words):
        return all(self.contains(w) for w in words)
