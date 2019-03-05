import os
import ujson

import gensim
import numpy as np

import linear_regression_alignment
from models_manager import AlignmentMethod, ModelsManager
from word2vec_specific_model import Word2VecSpecificModel
from word2vec_wiki_model import Word2VecWikiModel


def filter_model_vocab(wv, top, words=None):
    """
    Filters a given KeyedVectors gensim object, wv, that's sorted by frequency (descending).
    Only the 'top' number of words is kept.
    If 'words' is set (as list or set), then this list is added to the vocabulary as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    The .vocab dictionary is also updated for wv, preserving the count but updating the index.
    """

    if words is None:
        words = []

    # note: we assume the wv is already sorted by frequency
    # sort by frequency
    # vocab = set(wv.vocab.keys())
    # vocab = list(vocab)
    # vocab.sort(key=lambda w: wv.vocab[w].count, reverse=True)

    # take the top words + words that were passed as a parameter
    new_index2word = [word for index, word in enumerate(wv.index2word) if index < top or word in words]

    # replace old syn0norm array with new one (with common vocab)
    indices = [wv.vocab[w].index for w in new_index2word]
    old_arr = wv.syn0norm if wv.syn0norm else wv.syn0
    new_arr = np.array([old_arr[index] for index in indices])
    wv.syn0norm = wv.syn0 = new_arr

    # replace old vocab dictionary and index2word with new ones
    wv.index2word = new_index2word
    old_vocab = wv.vocab
    new_vocab = {}
    for new_index, word in enumerate(new_index2word):
        old_vocab_obj = old_vocab[word]
        new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
    wv.vocab = new_vocab

    return wv


class GlobalToTempModel(object):
    def __init__(self, global_model, models_manager):
        self.global_model = global_model
        self.models_manager = models_manager
        self.alignment_method = AlignmentMethod.LinearRegression

    def convert_global_model_to_year(self, target_year):
        base_embed = self.global_model.wv
        target_embed = self.models_manager.year_to_model[target_year].wv
        if self.alignment_method == AlignmentMethod.LinearRegression:
            aligned_embed = linear_regression_alignment.align_embeddings(base_embed, target_embed)
        else:
            return None
        aligned_model = Word2VecSpecificModel(model=aligned_embed)
        return aligned_model


if __name__ == '__main__':
    min_year = 1981
    max_year = 2015
    title_id_map = ujson.load(open('data/title_id_map.json', encoding='utf-8'))
    event_ids = ujson.load(open('data/event_ids_since1980.json', encoding='utf-8'))
    event_ids = [str(eid) for eid in event_ids]
    global_model = Word2VecWikiModel('data/WikipediaClean5Negative300Skip10/WikipediaClean5Negative300Skip10',
                                     title_id_map)
    filter_model_vocab(global_model.wv, top=100000, words=event_ids)
    nyt_models_dir = r'data/models_wiki_sg_win5_size140_min50_remstopwords'
    models_manager = ModelsManager(nyt_models_dir, from_year=min_year, to_year=max_year)
    models_manager.load_models()
    converter = GlobalToTempModel(global_model.wv, models_manager)
    for year in range(min_year, max_year + 1):
        new_model = converter.convert_global_model_to_year(year)
        # save the new model with precomputed L2-normalized vectors
        new_model.model.init_sims(replace=True)
        file_name = 'word2vec-nyt-%i.model' % year
        new_model.model.save(os.path.join(nyt_models_dir, 'converted_from_wiki', file_name))
