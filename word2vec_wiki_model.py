from gensim.models import KeyedVectors

from word2vec_model import Word2VecModelBase


class Word2VecWikiModel(Word2VecModelBase):
    def __init__(self, model_path, title_id_map, id_title_map=None):
        self.wv = KeyedVectors.load(model_path).wv
        self.id_title_map = id_title_map
        self.title_id_map = title_id_map

    def get_word_vector(self, word):
        id = self.get_word_or_index(word)
        if id is not None:
            return self.wv[id]
        return None

    def contains(self, word):
        return self.get_word_or_index(word) is not None

    def get_word_or_index(self, word):
        """
        If the given word exists in the model, returns it. Otherwise, if it's a Wikipedia title, returns its ID.
        :param word:
        :return:
        """
        word = str(word)
        if word in self.wv:
            return word
        elif word in self.title_id_map:  # check if 'word' is an entity - look for its id
            id = str(self.title_id_map[word])
            if id in self.wv:
                return id
        return None

    def similarity(self, word1, word2):
        return self.wv.similarity(self.get_word_or_index(word1), self.get_word_or_index(word2))

    def rank(self, word1, word2):
        return self.wv.rank(self.get_word_or_index(word1), self.get_word_or_index(word2))

    def get_word_by_index(self, index):
        word = self.wv.index2word[index]
        return self.convert_to_string(word)

    def convert_to_strings(self, words):
        for word in words:
            yield self.convert_to_string(word)

    def convert_to_string(self, word):
        if self.id_title_map and word in self.id_title_map:  # check if 'word' is an entity (id)
            word = self.id_title_map[word]
        return word

    def contains_all_words(self, words):
        return all(self.contains(w) for w in words)
