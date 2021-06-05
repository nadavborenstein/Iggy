import numpy as np
import pandas as pd

PATH_TO_FUN_CSV = "resources/word-funnines-model/Why_are_some_words_funny_lemmas.csv"


class FunModel:
    """
    Funniness model. Given a word, the model returns a value describing
    how funny the word is (lower is better)
    """

    def __init__(self, type="Arousal"):
        """
        builds the funniness model
        :param type: the funniness measurement type. Default is "Arousal".
        """
        self._model = dict()
        self.type = type
        data = pd.read_csv(PATH_TO_FUN_CSV)
        for i, row in data.iterrows():
            self._model[row[0]] = float(row[type])
        self._default_value = np.mean(list(self._model.values()))

    def __getitem__(self, word):
        """
        Returns the AoA score of "word". If the word is not in the vocabulary the default value is returned
        """
        if word in self:
            return self._model[word]
        else:
            return self._default_value

    def __contains__(self, word):
        """
        Returns true if "word" is in the vocabulary, and false otherwise
        """
        return word in self._model
