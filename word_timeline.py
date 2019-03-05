import re
import ujson
from collections import defaultdict, OrderedDict

import numpy as np

from events_classifier import EventClassifier
from max_heap import MaxHeap
from models_manager import Method
from word2vec_wiki_model import Word2VecWikiModel

min_year = 1981
max_year = 2015
all_years = list(range(min_year, max_year + 1))


class WordOnthology(object):
    def __init__(self, models_manager, knn_threshold, w2v_threshold, num_of_neighbors, support_events=False,
                 global_model=None, transformed_temporal_models=False, limit_years_around=0):
        self.knn_threshold = knn_threshold
        self.w2v_threshold = w2v_threshold
        self.models_manager = models_manager
        self.num_of_neighbors = num_of_neighbors
        event_year = ujson.load(open('data/event_year_since1980.json', encoding='utf-8'))
        self.event_to_content = ujson.load(open('data/event_content_since1980.json', encoding='utf-8'))
        self.event_to_text_content = ujson.load(open('data/event_text_content_since1980.json', encoding='utf-8'))
        self.year_to_event = defaultdict(list)
        for event, year in event_year.items():
            self.year_to_event[year].append(event)
        self.support_events = support_events
        self.limit_years_around = limit_years_around
        if support_events:
            if transformed_temporal_models:
                self.classifier = EventClassifier(models_manager=self.models_manager)
            else:
                self.classifier = EventClassifier(global_model=global_model)
            self.classifier.create_classifier(train=True)
        self.global_model_inner = global_model
        self.transformed_temporal_models = transformed_temporal_models

    def get_similar_words_per_year(self, word):
        if not word:
            return None
        year_to_similar_words = OrderedDict()
        for year in range(min_year, max_year):
            similar_words = self.models_manager.most_similar_words_in_year(word, year, self.num_of_neighbors)
            year_to_similar_words[year] = similar_words
        return year_to_similar_words

    def find_key_years(self, word, method):
        """
        find key years of a given word, according to a given method (e.g. KNN, word2vec)
        """
        if not word:
            return None
        method_threshold = self.knn_threshold if method == Method.KNN else self.w2v_threshold
        year_to_sim, peaks = self.models_manager.get_scores_peaks(word, min_year, max_year, method,
                                                                  threshold=method_threshold, k=self.num_of_neighbors)
        return peaks

    def find_key_events_by_classifier(self, word, min_classifier_score, max_events_per_year,
                                      existing_key_years_to_events, include_score=False):
        """
        find important events using our events classifier, and word2vec similarities as a filter.
        'key_years_to_events' should be calculated by another method ('find_key_events_...'),
        preferably with a bigger max_events_num, as we don't want to just filter an existing method.
        """
        if not word:
            return None
        word = word.lower()
        key_years_to_events = OrderedDict([(year, []) for year in all_years])
        for key_year, top_events_scores in existing_key_years_to_events.items():
            if not top_events_scores:
                continue
            # run the classifier for these events
            event_to_features = {}
            event_to_prev_method_score = {}
            for event, score in top_events_scores:
                event_to_prev_method_score[event] = float(score)
                feature_vector, feature_names = self.classifier.featurize_event_word((event, word))
                if feature_vector is not None:
                    event_to_features[event] = feature_vector
            probs = list(self.classifier.classifier.classifier.predict_proba(
                list(event_to_features.values())))  # probabilities for the true class
            y_prob = np.array(probs)[:, 1]
            top_key_events = MaxHeap(max_events_per_year)
            for event_i, event in enumerate(list(event_to_features.keys())):
                event_score = (y_prob[event_i] * 4 + event_to_prev_method_score[event] * 6) / 10
                top_key_events.add(event_score, event)
            top_key_events = sorted(top_key_events.heap, reverse=True)
            key_years_to_events[key_year] = [item[1] + '--' + str(round(item[0], 2)) if include_score else item[1] for
                                             item in top_key_events if item[0] > min_classifier_score]
        return key_years_to_events

    def find_key_events_by_knn(self, word, max_events_per_year, years, include_score=False):
        """
        find  events that are closest to the given word and its nearest neighbors
        """
        if not word:
            return None
        word = word.lower()
        year_to_similar_words = self.get_similar_words_per_year(word)
        key_years_to_events = OrderedDict([(year, []) for year in years])

        for key_year in years:
            model = self.get_model(key_year)
            # find the key events from that year
            top_key_events = MaxHeap(max_events_per_year)
            # take the events that are most similar to the KNN
            word_knn = [word] + year_to_similar_words[key_year] if year_to_similar_words[key_year] is not None else [
                word]
            events = self.get_relevant_events(key_year)
            for e in events:
                knn_similarities = [model.similarity(e, sim_word) for sim_word in word_knn
                                    if word in self.event_to_content[e] and model.contains_all_words([e, sim_word])]
                if len(knn_similarities) > 0:
                    similarity = np.mean(knn_similarities)
                    if similarity > self.knn_threshold:
                        top_key_events.add(similarity, e)
            top_key_events = sorted(top_key_events.heap, reverse=True)
            key_years_to_events[key_year] = [(item[1], str(round(item[0], 2))) if include_score else item[1] for
                                             item in top_key_events]

        return key_years_to_events

    def find_key_events_by_word(self, word, max_events_per_year, years, include_score=False):
        """
        find events closest to the given word
        """
        if not word:
            return None
        word = word.lower()
        key_years_to_events = OrderedDict([(year, []) for year in years])
        for key_year in years:
            model = self.get_model(key_year)
            # find the key events from that year
            top_key_events = MaxHeap(max_events_per_year)
            # take the events that are most similar to the event
            events = self.get_relevant_events(key_year)
            for e in events:
                if word in self.event_to_content[e] and model.contains_all_words([e, word]):
                    similarity = model.similarity(e, word)
                    if similarity > self.knn_threshold:
                        top_key_events.add(similarity, e)
            top_key_events = sorted(top_key_events.heap, reverse=True)
            key_years_to_events[key_year] = [item[1] + '--' + str(round(item[0], 2)) if include_score else item[1] for
                                             item in top_key_events]

        return key_years_to_events

    def find_new_words_knn(self, word, years):
        """
        find for each given year: words that were added since the previous year
        """
        if not word:
            return None
        word = word.lower()
        year_to_similar_words = self.get_similar_words_per_year(word)
        year_to_new_words = OrderedDict()
        prev_similar_words = None
        for year in years:
            similar_words = year_to_similar_words[year]
            if prev_similar_words and similar_words is not None:  # mark new words
                year_to_new_words[year] = [w for w in similar_words if w not in prev_similar_words]
            else:
                year_to_new_words[year] = []
            prev_similar_words = similar_words
        return year_to_new_words

    def find_events_from_wikipedia_baseline(self, word, max_events_per_year, years, include_score=False,
                                            min_occurrences=5):
        """
        find for each given year: events that contain the given word the most times
        """
        if not word:
            return None
        word = word.lower()
        key_years_to_events = OrderedDict([(year, []) for year in years])
        for key_year in years:
            # find the key events from that year
            top_key_events = MaxHeap(max_events_per_year)
            # take the events that are most similar to the event
            for e in self.year_to_event[key_year]:
                # count number of occurrences of the given word in the Wiki content
                score = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), self.event_to_text_content[e].lower()))
                if score > min_occurrences:
                    top_key_events.add(score, e)
            top_key_events = sorted(top_key_events.heap, reverse=True)
            key_years_to_events[key_year] = [item[1] + '--' + str(round(item[0], 2)) if include_score else item[1] for
                                             item in top_key_events]

        return key_years_to_events

    def get_relevant_events(self, year):
        relevant_years = [y for y in range(year, year + self.limit_years_around + 1)] + [y for y in range(
            year - self.limit_years_around, year)]
        return [event for year, events in self.year_to_event.items() for event in events if year in relevant_years]

    def get_model(self, year=None):
        """
        returns the temporal model if we're using transformed models (o.w. they won't contain events)
        :param year:
        :return:
        """
        if year and self.transformed_temporal_models:
            return self.models_manager.get_model(year)
        else:
            return self.global_model

    @property
    def global_model(self):
        if not self.global_model_inner:
            title_id_map = ujson.load(open('data/title_id_map.json', encoding='utf-8'))
            self.global_model_inner = Word2VecWikiModel(
                'data/WikipediaClean5Negative300Skip10/WikipediaClean5Negative300Skip10',
                title_id_map)
            if self.classifier and self.classifier.global_model is None:
                self.classifier.global_model = self.global_model_inner
        return self.global_model_inner
