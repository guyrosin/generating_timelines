import re
from collections import OrderedDict
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine
import spacy

import linear_regression_alignment
from word2vec_specific_model import Word2VecSpecificModel
from word2vec_specific_wiki_model import Word2VecSpecificWikiModel
import utils
import peak_detection


class AlignmentMethod(utils.AutoNumber):
    NoAlign = ()
    Procrustes = ()
    LinearRegression = ()


class Method(utils.AutoNumber):
    No = ()
    KNN = ()
    W2V = ()
    Local = ()


class ModelsManager(object):
    def __init__(self, data_dir_name=sys.path[0], from_year=None, to_year=None, global_model_dir=None,
                 alignment_method=None, title_id_map=None, id_title_map=None):
        """
        :param data_dir_name:
        :param from_year:
        :param to_year:
        :param global_model_dir: directory of the global model (leave blank if it's in the same one)
        """
        self.data_dir_name = data_dir_name
        self.year_to_model_inner = {}
        self.from_year = from_year if from_year else 0
        self.to_year = to_year if to_year else 9999
        self.global_model_dir = global_model_dir
        self.aligned_models = {}
        if alignment_method:
            self.alignment_method = alignment_method
        self.title_id_map = title_id_map
        self.id_title_map = id_title_map
        self.nlp_inner = None

    def build_model_of_year(self, year, file=None):
        if self.title_id_map:
            self.year_to_model_inner[year] = Word2VecSpecificWikiModel(year, self.data_dir_name, file,
                                                                       title_id_map=self.title_id_map,
                                                                       id_title_map=self.id_title_map)
        else:
            self.year_to_model_inner[year] = Word2VecSpecificModel(year, self.data_dir_name, file)

    def build_models_from_files(self, files):
        with ThreadPoolExecutor(max_workers=4) as executor:  # this will wait until completion of all workers
            for f in files:
                # look for years in the filename
                m = re.match(r'.*nyt-(\d{4}).*$', f)
                if m is None:
                    continue
                year = int(m.group(1))
                if year not in self.year_to_model_inner and self.from_year <= year <= self.to_year:
                    logging.info('Building model for %i', year)
                    executor.submit(self.build_model_of_year, year, f)
            if 9999 not in self.year_to_model_inner and self.global_model_dir is not None:
                self.year_to_model_inner[9999] = Word2VecSpecificModel(9999, dir_path=self.global_model_dir)

    def contains_year(self, year):
        return year in self.year_to_model.keys()

    def get_model(self, year):
        return self.year_to_model[year] if year in self.year_to_model else None

    def load_models(self):
        # build a word2vec model out of each model file in the 'data' folder
        files_list = os.listdir(self.data_dir_name)
        files_list = filter(lambda file: file.endswith('.model'), files_list)
        self.build_models_from_files(files_list)

        # convert to an OrderedDict, sorted by key
        self.year_to_model_inner = OrderedDict(sorted(self.year_to_model_inner.items(), key=lambda t: t[0]))
        return self.year_to_model_inner

    def get_aligned_model(self, source_year, target_year, alignment_method):
        if alignment_method is None or alignment_method == AlignmentMethod.NoAlign:
            return self.get_model(target_year)

        key = (source_year, target_year)
        if key in self.aligned_models:
            return self.aligned_models[key]

        base_embed = self.year_to_model[source_year].wv
        target_embed = self.year_to_model[target_year].wv
        if alignment_method == AlignmentMethod.LinearRegression:
            aligned_embed = linear_regression_alignment.align_embeddings(base_embed, target_embed)
        else:
            return None
        aligned_model = Word2VecSpecificModel(model=aligned_embed)
        self.aligned_models[key] = aligned_model
        return aligned_model

    def calc_changed_score(self, word, before_year, after_year, similarity_metric):
        if not self.contains_year(before_year) or not self.contains_year(after_year):
            # logging.warning('year out of range')
            return None
        start_model = self.get_model(before_year)
        end_model = self.get_aligned_model(before_year, after_year, self.alignment_method)
        if not start_model or not end_model:
            return None
        if start_model.contains(word) and end_model.contains(word):
            w1 = start_model.get_word_vector(word)
            w2 = end_model.get_word_vector(word)
            if similarity_metric == cosine:
                dst = 1 - similarity_metric(w1, w2)
            else:
                dst = similarity_metric(w1, w2) / 2  # 2 is the max euclidean distance between two normalized vectors
            return dst
        else:
            # logging.warning("word doesn't exist in vocabulary")
            return None

    def changed_score_w2v(self, word, year, similarity_metric=cosine):
        """
        compare the word2vec representations of the word in 2 consecutive years
        """
        year_prev = year - 1
        if year_prev < self.from_year or year > self.to_year:
            return None
        return self.calc_changed_score(word, year_prev, year, similarity_metric)

    def changed_score_knn(self, word, year, k=25):
        """
        look at the k nearest neighbors at the given year and measure how many changed compared to the previous year
        """
        year_prev = year - 1
        if year_prev < self.from_year or year > self.to_year:
            return None
        try:
            nn_before = self.most_similar_words_in_year(word, year_prev, k=k)
            nn_after = self.most_similar_words_in_year(word, year, k=k)
            intersection = set(nn_before).intersection(nn_after)
        except (KeyError, TypeError):  # the word doesn't exist in one of the models
            return None
        return (k - len(intersection)) / k

    def changed_score_local(self, word, year, k=25):
        """
        Basic implementation of William Hamilton (@williamleif) et al's measure of semantic change
        proposed in their paper "Cultural Shift or Linguistic Drift?" (https://arxiv.org/abs/1606.02821),
        which they call the "local neighborhood measure." They find this measure better suited to understand
        the semantic change of nouns owing to "cultural shift," or changes in meaning "local" to that word,
        rather than global changes in language ("linguistic drift") use that are better suited to a
        Procrustes-alignment method (also described in the same paper.)
        """
        year_prev = year - 1
        if year_prev < self.from_year or year > self.to_year:
            return None
        try:
            nn_before = self.most_similar_words_in_year(word, year_prev, k=k)
            nn_after = self.most_similar_words_in_year(word, year, k=k)
            intersection = set(nn_before).intersection(nn_after)
            # for both models, get a similarity vector between the focus word and all of the neighbors
            vector1 = [self.year_to_model[year_prev].similarity(word, w) for w in intersection]
            vector2 = [self.year_to_model[year].similarity(word, w) for w in intersection]
        except (KeyError, TypeError):  # the word doesn't exist in one of the models
            return None
        # compute the cosine distance *between* those similarity vectors
        dist = cosine(vector1, vector2)
        # return this cosine distance -- a measure of the relative semantic shift for this word between these two models
        return dist

    def get_scores_peaks(self, word, min_year, max_year, method, peak_detection_method=peak_detection.find_peaks,
                         threshold=None, k=20):
        if not word:
            return None
        year_to_sim = OrderedDict([(year, 0) for year in range(min_year + 1, max_year)])
        for year in year_to_sim.keys():
            if method == Method.KNN:
                score = self.changed_score_knn(word, year, k)
            elif method == Method.W2V:
                score = self.changed_score_w2v(word, year)
            else:
                score = self.changed_score_local(word, year)
            if score:
                year_to_sim[year] = score
        peak_years = peak_detection_method(year_to_sim).astype(int)
        if threshold:
            peak_years = set(peak_years)
            peak_years.update([year for year, sim in year_to_sim.items() if sim >= threshold])
            peak_years = sorted(peak_years)
        return year_to_sim, peak_years

    def most_similar_words_in_year(self, word, year, k=10, lemmatize=True):
        model = self.get_model(year)
        if model and model.contains(word):
            similar_words = [w for w, p in model.wv.most_similar(word, topn=k)]
            if lemmatize:  # lemmatize to remove duplicate words
                similar_words = [self.nlp(word)[0].lemma_ for word in similar_words]
                similar_words = list(OrderedDict.fromkeys(similar_words))  # remove duplicates
                total_top = k
                while len(similar_words) < k:
                    # take additional similar words instead of the duplicates
                    len_diff = k - len(similar_words)
                    total_top += len_diff
                    additional_words = [w for w, p in
                                        model.wv.most_similar(word, topn=total_top)[:-len_diff]]
                    similar_words.extend([self.nlp(word)[0].lemma_ for word in additional_words])
                    similar_words = list(OrderedDict.fromkeys(similar_words))  # remove duplicates
            similar_words = list(self.convert_to_strings(similar_words))
            return similar_words

    def convert_to_strings(self, words):
        for word in words:
            yield self.convert_to_string(word)

    def convert_to_string(self, word):
        if self.id_title_map and word in self.id_title_map:  # check if 'word' is an entity (id)
            word = self.id_title_map[word]
        return word

    @property
    def year_to_model(self):
        if not self.year_to_model_inner:
            self.load_models()
        return self.year_to_model_inner

    @property
    def nlp(self):
        if not self.nlp_inner:
            self.nlp_inner = spacy.load('en')
        return self.nlp_inner
