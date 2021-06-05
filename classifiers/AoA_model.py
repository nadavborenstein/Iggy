import pandas as pd
import numpy as np

PATH_TO_AOA_CSV = "resources/AoA/AoA_lemmas.csv"


class AoAModel:
    """
    AoA (Age of Acquisition) model. Given a word, the model returns a value describing
    how simple the word is (lower is simpler)
    """

    def __init__(self, default_value=16):
        """
        builds the AoA model
        :param default_value: default value of the model (16 is the maximal possible value)
        """
        self._model = dict()

        data = pd.read_csv(PATH_TO_AOA_CSV)
        for i, row in data.iterrows():
            if not pd.isna(row.AoArating):
                self._model[row[0]] = float(row.AoArating)
            else:
                self._model[row[0]] = float(row.LWV)

        self._default_value = (
            default_value
            if default_value is not None
            else np.mean(list(self._model.values()))
        )

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
