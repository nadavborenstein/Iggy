import pickle
import numpy as np

UNIGRAMS_PATH = "resources/ngram-language-models/{}_unigrams.p"
BIGRAMS_PATH = "resources/ngram-language-models/{}_bigrams.p"
TRIGRAMS_PATH = "resources/ngram-language-models/{}_trigrams.p"

POS_UNIGRAMS_PATH = "resources/ngram-language-models/{}_unigrams_pos_tags.p"
POS_BIGRAMS_PATH = "resources/ngram-language-models/{}_bigrams_pos_tags.p"
POS_TRIGRAMS_PATH = "resources/ngram-language-models/{}_trigrams_pos_tags.p"


class NgramBasedLM(object):
    """
    An N-gram based language model. Supports 1-gram, 2-gram and 3-gram LMs as well as N-gram LMs
    based on pos tags.
    """

    def __init__(self, type: str, grams: int = 2, pos_tags: bool = False):
        """
        Init the N-gram language model
        :param type: The type of the model. can be either "science" or "jokes".
        :param grams: one of  the values in [1,2,3]
        :param pos_tags: Whether to use a pos tags based N-gram model or not
        """
        self.type = type
        self.grams = grams

        # self.trigrams, self.bigrams and self.unigrams are all default dicts
        if pos_tags:
            self.unigrams = pickle.load(open(POS_UNIGRAMS_PATH.format(type), "rb"))
            self.bigrams = (
                pickle.load(open(POS_BIGRAMS_PATH.format(type), "rb"))
                if grams >= 2
                else None
            )
            self.trigrams = (
                pickle.load(open(POS_TRIGRAMS_PATH.format(type), "rb"))
                if grams == 3
                else None
            )
        else:
            self.unigrams = pickle.load(open(UNIGRAMS_PATH.format(type), "rb"))
            self.bigrams = (
                pickle.load(open(BIGRAMS_PATH.format(type), "rb"))
                if grams >= 2
                else None
            )
            self.trigrams = (
                pickle.load(open(TRIGRAMS_PATH.format(type), "rb"))
                if grams == 3
                else None
            )
        self.unigrams_total = sum(self.unigrams.values())

    def __getitem__(self, ngram: tuple) -> float:
        """
        :param ngram: a tuple of words of length 1, 2 or 3.
        :return: the probability of seeing the last word in the tuple given the other words
        """
        if self.grams == 3:
            w_2, w_1, w = ngram
            trigram_count = self.trigrams[(w_2, w_1, w)]
            w_1_w_2_bigram_count = self.bigrams[(w_2, w_1)]
            bigrams_count = self.bigrams[(w_1, w)]
            w_1_unigram_count = self.unigrams[w_1]
        elif self.grams == 2:
            w_1, w = ngram
            bigrams_count = self.bigrams[(w_1, w)]
            w_1_unigram_count = self.unigrams[w_1]
            trigram_count, w_1_w_2_bigram_count = 0, 0
        else:
            w = ngram[0]
            trigram_count, w_1_w_2_bigram_count, bigrams_count, w_1_unigram_count = (
                0,
                0,
                0,
                0,
            )
        # the actual calculation, implementation of simple smoothing technique
        w_unigram_count = self.unigrams[w]
        if trigram_count != 0:
            return trigram_count / w_1_w_2_bigram_count
        elif bigrams_count != 0:
            return bigrams_count / w_1_unigram_count
        else:
            return max(w_unigram_count / self.unigrams_total, 1 / self.unigrams_total)

    def _assert_sentence(self, sent_as_ngram: list) -> None:
        """
        Make sure that the sentence N-gram N matches the N of the model
        """
        assert (
            len(sent_as_ngram[0]) == self.grams
        ), f"Sentence N-grams are different from the model N-grams ({len(sent_as_ngram[0])} vs. {self.grams})."

    def sentence_entropy(self, sent_as_ngram: list) -> float:
        """
        Calculates the entropy of a sentence
        :param sent_as_ngram: The sentence already converted to N-grams
        """
        self._assert_sentence(sent_as_ngram)
        entropy = np.mean([np.log2(self[ngram]) for ngram in sent_as_ngram])
        return float(entropy)

    def sentence_entropy_custom_reduce(
        self, sent_as_ngram: list, reduce=np.mean
    ) -> float:
        """
        Compute sentence "entropy" while using a custom reduction (this is the same as regular
        entropy if reduce=mean).
        :param sent_as_ngram: The sentence already converted to N-grams
        :param reduce: the custom reduce ti use. Default is np.mean.
        """
        self._assert_sentence(sent_as_ngram)
        entropy = reduce([np.log2(self[ngram]) for ngram in sent_as_ngram])
        return float(entropy)
