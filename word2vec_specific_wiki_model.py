import os

from word2vec_wiki_model import Word2VecWikiModel


class Word2VecSpecificWikiModel(Word2VecWikiModel):
    def __init__(self, year=None, dir_path=None, file=None, model=None, title_id_map=None, id_title_map=None):
        if model:
            self.model = model
            return
        if not file and year:
            file = 'word2vec-nyt-%i.model' % year
        model_path = os.path.join(dir_path, file)
        super().__init__(model_path, title_id_map, id_title_map)
