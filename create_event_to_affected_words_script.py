import csv
import heapq
import logging

import rapidjson as json
from scipy.spatial import distance

from max_heap import MaxHeap
from models_manager import ModelsManager, AlignmentMethod, SpecificModelType
from word2vec_wiki2vec_model import Word2VecWiki2VecModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


WIKI2VEC_PATH = 'data/wikipedia2vec/wiki2vec.kv'
TEMPORAL_MODELS_PATH = 'data/nyt_models'



def generate_negatives_from_same_time(related_words, num_of_words, year, models_manager, distance_metric,
                                      affected_words=None):
    """
    find the words that changed the least during the given year
    """
    before_year = year - 1
    after_year = year + 1
    if before_year < models_manager.from_year or after_year > models_manager.to_year:
        return None
    score_word_heap = []
    # for every word (in the close words subset), calculate how much it changed before & after the event
    for related_word in related_words:
        if affected_words and related_word in affected_words:  # skip affected (positive) words
            continue
        score = models_manager.calc_changed_score(related_word, before_year, after_year, distance_metric)
        if not score:
            continue
        score_word_heap.append((score, related_word))
    heapq.heapify(score_word_heap)
    words = [item[1] for item in heapq.nsmallest(num_of_words, score_word_heap)]
    return words


def filter_related_words_by_similarity(event, words, global_model, threshold):
    if event not in global_model:
        return []
    related_words_filtered = []
    existing_words_count = 0
    for related_word in words:
        if related_word in global_model:
            existing_words_count += 1
            similarity = global_model.similarity(related_word, event)
            if similarity is not None and similarity > threshold:
                related_words_filtered.append(related_word)
    logging.info(f'filtered {len(related_words_filtered)} / {existing_words_count}')
    return related_words_filtered


def filter_affected_words(related_words, num_of_words, year, models_manager, distance_metric):
    """
    look for words that were changed in the time of the event, and were not changed before it
    """
    before_year = year - 1
    before2_year = year - 2
    after_year = year + 1
    if before2_year < models_manager.from_year or after_year > models_manager.to_year:
        return None
    most_changed_words = MaxHeap(num_of_words)
    # for every word (in the close words subset), calculate how much it changed before & after the event
    for related_word in related_words:
        score = models_manager.calc_changed_score(related_word, before_year, after_year, distance_metric)
        if not score:
            continue
        before_score = models_manager.calc_changed_score(related_word, before2_year, before_year, distance_metric)
        if not before_score:
            continue
        total_score = score - before_score
        most_changed_words.add(total_score, related_word)
    words = [item[1] for item in most_changed_words.heap]
    return words


if __name__ == "__main__":
    num_of_words = 20
    distance_metric = distance.cosine
    alignment_method = AlignmentMethod.Procrustes
    min_semantic_sim = 0.4
    min_year = 1981
    max_year = 2018
    global_model = Word2VecWiki2VecModel(WIKI2VEC_PATH)
    event_id_year = json.load(open('data/event_id_year_since1980.json', encoding='utf-8'))
    event_id_name = json.load(open('data/event_id_name_since1980.json', encoding='utf-8'))
    event_name_id = json.load(open('data/event_name_id_since1980.json', encoding='utf-8'))
    event_id_to_top_tfidf = json.load(open('data/event_id_to_top_tfidf_100.json', encoding='utf-8'))
    event_ids = event_id_year.keys()
    event_to_affected_words_file = 'data/event_filtered_to_{}_affected_words.json'.format(
        num_of_words)
    event_to_unaffected_words_file = 'data/event_filtered_to_{}_unaffected_words.json'.format(
        num_of_words)
    event_word_dataset_file = 'data/events_words_dataset_{}affected.tsv'.format(num_of_words)
    models_manager = ModelsManager(TEMPORAL_MODELS_PATH,
                                   from_year=min_year, to_year=max_year, alignment_method=alignment_method,
                                   specific_model_type=SpecificModelType.Wiki2Vec)
    models_manager.load_models()
    event_to_affected_words = {}
    event_to_affected_words_other_reason = {}
    event_word_items = []
    i = 0
    for event_id, event_year in event_id_year.items():
        event_year = int(event_year)
        event = event_id_name[event_id]
        if event_year < min_year or event_year > max_year:
            continue
        if event_id not in event_id_to_top_tfidf:
            logging.warning(f'No content for {event}')
            continue
        # tokenized_content = event_content_tokenized[event]
        related_words = [word for word, score in event_id_to_top_tfidf[event_id].items()]
        related_words_filtered = filter_related_words_by_similarity(event, related_words,
                                                                    global_model, min_semantic_sim)
        if not related_words_filtered:
            continue
        affected_words = filter_affected_words(related_words_filtered, num_of_words, event_year, models_manager,
                                               distance_metric)
        if not affected_words:
            continue
        affected_words_other_reason = generate_negatives_from_same_time(related_words_filtered, num_of_words,
                                                                        event_year, models_manager, distance_metric,
                                                                        affected_words=affected_words)
        if not affected_words_other_reason:
            continue
        # make sure to create an even number of positive and negative examples
        min_size = min(len(affected_words), len(affected_words_other_reason))
        affected_words = affected_words[:min_size]
        affected_words_other_reason = affected_words_other_reason[:min_size]
        event_to_affected_words[event] = affected_words
        event_to_affected_words_other_reason[event] = affected_words_other_reason
        event_word_items.extend([event, word, 1] for word in affected_words)
        event_word_items.extend([event, word, 0] for word in affected_words_other_reason)
        i += 1
        logging.info(f'event #{i}')

    logging.info(f"writing to {event_to_affected_words_file} {len(event_to_affected_words)} events")
    with open(event_to_affected_words_file, 'w', encoding='utf-8') as outfile:
        json.dump(event_to_affected_words, outfile, indent=2)
    with open(event_to_unaffected_words_file, 'w', encoding='utf-8') as outfile:
        json.dump(event_to_affected_words_other_reason, outfile, indent=2)
    with open(event_word_dataset_file, 'w', encoding='utf-8', newline='') as out_file:
        out = csv.writer(out_file)
        for item in event_word_items:
            out.writerow(item)
