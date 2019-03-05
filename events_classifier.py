import csv
import logging
import os
from collections import OrderedDict
import pickle
import numpy as np
import ujson
import pandas as pd
from gensim import corpora

from classifier import Classifier

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
This classifier receives an event and a word, and outputs a score of how likely this event changed the word.
It can use either a single global model (i.e. from Wikipedia) or multiple temporal models (through a ModelsManager object)
"""


class EventClassifier(object):
    def __init__(self, event_bow=False, global_model=None, models_manager=None, print_results=False,
                 classifier_type='rf', load_existing_features=False):
        self.similarity_feature = True
        self.global_model = global_model
        self.models_manager = models_manager
        self.classifier_type = classifier_type
        self.features_file_name = 'data/events_features.pickle'
        self.load_existing_features = load_existing_features and os.path.exists(self.features_file_name)
        self.classifier = None
        self.print_results = print_results
        if not load_existing_features:
            self.event_types_dict = None
            # self.event_types_dict = get_types_feature()
            # self.event_impact_dict = None
            self.event_to_link_count = ujson.load(open('data/event_link_count_norm_since1980.json', encoding='utf-8'))
            # self.event_to_pageviews = None
            self.event_to_pageviews = ujson.load(open('data/event_to_pageviews_norm_since1980.json', encoding='utf-8'))
            # self.events_refs_num_map = None
            self.events_refs_num_map = ujson.load(open('data/event_to_refs_num_norm_since1980.json', encoding='utf-8'))
            self.event_year = ujson.load(open('data/event_year_since1980.json', encoding='utf-8'))
            self.events = None
            self.event_to_bow = None
            if event_bow:
                self.bow_vocab = None
                self.load_event_to_bow()

    @staticmethod
    def read_event_word_dataset(file_name):
        """
        Reads through a dataset of event,word pairs and creates a feature vector.
        :param file_name:
        :return:
        """

        dataset = OrderedDict()
        with open(file_name, encoding='utf8') as relations_file:
            rel_reader = csv.reader(relations_file)
            for rel in rel_reader:
                # each line contains: event, word, label
                event = rel[0]
                word = rel[1]
                # parse the label
                label = int(rel[-1])
                dataset[event, word] = label

        return dataset

    @staticmethod
    def get_types_feature():
        event_types_df = pd.read_csv('data/df_event_name_types_since1980_800views.csv', index_col=0)
        event_types_df.fillna(0, inplace=True)
        return event_types_df.T.to_dict('list')

    def get_model(self, year=None):
        # return the temporal model if we're using transformed models (o.w. they won't contain events)
        if year and self.models_manager:
            return self.models_manager.get_model(year)
        else:
            return self.global_model

    def create_classifier(self, train=False):
        load_existing_model = False
        model_file_name = 'data/model.pickle'
        if load_existing_model and os.path.exists(model_file_name):
            logging.info('loading existing model file: {}'.format(model_file_name))
            self.classifier = pickle.load(open(model_file_name, 'rb'))
        else:
            if self.load_existing_features:
                logging.info('loading existing features file: {}'.format(self.features_file_name))
                features_dict, feature_names = pickle.load(open(self.features_file_name, 'rb'))
            else:
                event_word_dataset_file = 'data/events_words_dataset_20affected.tsv'
                dataset = self.read_event_word_dataset(event_word_dataset_file)

                features_dict = OrderedDict()
                for event_word, label in dataset.items():
                    feature_vec, feature_names = self.featurize_event_word(event_word, label)
                    if feature_vec is not None:
                        features_dict[event_word] = feature_vec
                pickle.dump((features_dict, feature_names), open(self.features_file_name, 'wb'))
            logging.info('got {} items in the dataset'.format(len(features_dict)))
            feature_vectors = list(features_dict.values())
            feature_vectors = np.array(feature_vectors)
            item_names = list(features_dict.keys())
            X = feature_vectors[:, :-1].astype(float)
            y = feature_vectors[:, -1].astype(float)
            self.classifier = Classifier(X, y, self.classifier_type, feature_names=feature_names,
                                         print_results=self.print_results, item_names=item_names)
            if train:
                self.classifier.train()
            pickle.dump(self.classifier, open(model_file_name, 'wb'))
        return self.classifier

    def featurize_word(self, word, year=None):
        word_vec = self.get_model(year).get_word_vector(word)
        if word_vec is None:
            return None, None
        feature_names = ['word_emb_{}'.format(i) for i in range(len(word_vec))]
        feature_vec = word_vec
        return feature_vec, feature_names

    def featurize_event(self, event, year=None):
        feature_names = []
        if self.event_to_bow:
            event_vec = self.event_to_bow[event]
            if event_vec is not None:
                feature_names.extend(['event_emb_{}'.format(w) for w in self.bow_vocab])
            else:
                # logging.warning('no vector for {}'.format(event))
                pass
        else:
            event_vec = self.get_model(year).get_word_vector(event)
            if event_vec is not None:
                feature_names.extend(['event_emb_{}'.format(i) for i in range(len(event_vec))])
            else:
                # logging.warning('no vector for {}'.format(event))
                pass
        if event_vec is None:
            return None, None
        feature_vec = event_vec
        if self.event_types_dict:
            if event not in self.event_types_dict:
                logging.warning('no type for {}'.format(event))
                return None, None
            feature_vec = np.append(feature_vec, self.event_types_dict[event])
            feature_names.extend(['event_type_{}'.format(i) for i in range(len(self.event_types_dict[event]))])
        if self.event_to_link_count:
            if event not in self.event_to_link_count:
                logging.warning('no impact score for {}'.format(event))
                return None, None
            feature_vec = np.append(feature_vec, self.event_to_link_count[event])
            feature_names.append('event_link_count')
        if self.events_refs_num_map:
            if event not in self.events_refs_num_map:
                logging.warning('no refs num for {}'.format(event))
                return None, None
            feature_vec = np.append(feature_vec, self.events_refs_num_map[event])
            feature_names.append('event_refs_num')
        if self.event_to_pageviews:
            if event not in self.event_to_pageviews:
                logging.warning('no pageviews for {}'.format(event))
                return None, None
            feature_vec = np.append(feature_vec, self.event_to_pageviews[event])
            feature_names.append('event_pageviews')
        return feature_vec, feature_names

    def featurize_event_word(self, event_word, label=None):
        event = event_word[0]
        word = event_word[1]
        feature_names = []
        year = self.event_year[event] if self.models_manager else None
        event_vec, event_feature_names = self.featurize_event(event, year)
        word_vec, word_feature_names = self.featurize_word(word, year)
        if event_vec is None or word_vec is None:
            return None, None
        feature_vec = np.append(event_vec, word_vec)
        feature_names.extend(event_feature_names)
        feature_names.extend(word_feature_names)
        if self.similarity_feature:
            similarity = self.get_model(year).similarity(event, word)
            feature_vec = np.append(feature_vec, similarity)
            feature_names.append('similarity')
        if label is not None:
            feature_vec = np.append(feature_vec, label)
        if feature_vec.size:
            return feature_vec, feature_names
        return None, None

    def evaluate(self):
        if self.classifier is None:
            self.create_classifier()
        self.classifier.evaluate()

    def load_event_to_bow(self):
        dictionary = corpora.Dictionary.load('data/events_vocab.dict')
        self.bow_vocab = list(dictionary.values())
        corpus = corpora.MmCorpus('data/events_bow.mm')
        self.events = ujson.load(open('data/events_since1980.json', encoding='utf-8'))
        self.event_to_bow = OrderedDict()
        for i, event in enumerate(self.events):
            bow = corpus[i]  # [(0, 1), (2, 1)]
            bow_vec = np.zeros(len(dictionary.token2id))
            indices = [x[0] for x in bow]
            values = [x[1] for x in bow]
            bow_vec.put(indices, values)
            self.event_to_bow[event] = bow_vec
