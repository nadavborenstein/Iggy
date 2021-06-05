import pickle
import textstat

import numpy as np
import pandas as pd
from nltk import ngrams

from AoA_model import AoAModel
from funniness_model import FunModel
from ngram_based_lm import NgramBasedLM
from transformers_lm import GPT2LM, BERT_LM
from nbsvm_classifier import NbSvmClassifier

aoa_model = AoAModel()
fun_model = FunModel()


################ AoA functions #####################


def simple_aoa_analysis(row: pd.Series, reduce=np.mean) -> float:
    """
    Analyze the sentence using an AoA (Age of Acquisition) model.
    :param row: Row of a titles dataframe
    :param reduce: How to aggregate the results of a token-level statistics
    (default is taking the mean across the sentence, other options are min, max and std)
    :return: The aggregated result
    """
    mean_aoa = float(reduce([aoa_model[toc.lemma_.lower()] for toc in row["nlp"]]))
    return mean_aoa


################ simple str functions #####################


def len_analysis(row: pd.Series) -> int:
    """
    Returns the length of the sentence.
    """
    return len(row["nlp"])


def average_word_len(row: pd.Series, reduce=np.mean) -> float:
    """
    Calculates statistics of the length of the individual tokens of the sentence.
    """
    mean_word_len = float(reduce([len(toc.text) for toc in row["nlp"]]))
    return mean_word_len


################ readability #####################


class Readability(object):
    """
    Calculates the readability score of the sentence using different measurements (using
    the textstat package)
    """

    default_measurement = "automated_readability_index"

    def __init__(self, measurement: str = default_measurement):
        self.measurement = measurement
        self.__name__ = (
            "readability_" + measurement
        )  # So to have different name for each readability measurement

    def __call__(self, row: pd.Series) -> float:
        # This is is equivalent to `textstat.type(row["title"])`, where type is a function name
        return textstat.__dict__[self.measurement](row["title"])


################ perplexity based functions ####################


class NgramEntropy(object):
    """
    Calculates the entropy of a sentence using a simple N-gram language model
    """

    def __init__(
        self,
        type: str = "science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        """
        :param type: The type of the model. can be either "science" or "jokes".
        :param grams: one of  the values in [1,2,3]
        :param pos_tags: Whether to use a pos tags based N-gram model or not
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = type + "_{}grams_entropy".format(grams)
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(type, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        try:
            entropy = self.lm_model.sentence_entropy_custom_reduce(
                title_as_ngrams, self.reduce
            )
            return float(entropy)
        except ValueError:  # one of the values in inf or -inf
            return 0


class SimpleGPT2LM(object):
    """
    Calculates the perplexity of a sentence using a GPT-2 language model
    """

    def __init__(self, model: str = "resources/finetuned-gpt2/checkpoint/"):
        """
        :param model: Path to saved checkpoint, or to a model name
        (one of the models supported by https://huggingface.co/).
        """
        self.model = GPT2LM(model)
        self.__name__ = "simple_gpt2_lm_after_fine_tuning"

    def __call__(self, row: pd.Series) -> float:
        return self.model.calc_perplexity(row.title.lower())


# TODO add caching mechanism
class GPT2LMCustomReduce(object):
    """
    Calculates the perplexity of a sentence using a GPT-2 language model. Supports custom reduce function
    """

    def __init__(
        self, model: str = "resources/finetuned-gpt2/checkpoint/", reduce=np.mean
    ):
        """

        :param model: Path to saved checkpoint, or to a model name
        (one of the models supported by https://huggingface.co/).
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.model = GPT2LM(model)
        self.__name__ = "gpt2_lm_after_fine_tuning_reduce_" + reduce.__name__
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        return self.model.calc_perplexity_custom_reduce(
            row.title.lower(), reduce=self.reduce
        )


class SimpleBERT_LM(object):
    """
    Calculates the perplexity of a sentence using BERT language model.
    """

    models = ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]

    def __init__(self, model: str = "bert-base-uncased"):
        """
        :param model: The BERT model to use. One of ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]
        """
        self.model = BERT_LM(model)
        self.model_name = model
        self.__name__ = f"simple_{model}_lm"

    def __call__(self, row: pd.Series) -> float:
        inp = row.title.lower() if "uncased" in self.model_name else row.title
        return self.model.calc_entire_sentence_perplexity(inp)


# TODO add caching mechanism
class BERT_LMCustomReduce(object):
    """
    Calculates the perplexity of a sentence using BERT language model. Supports custom reduce function
    """

    models = ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]

    def __init__(self, model: str = "bert-base-uncased", reduce=np.mean):
        """
        :param model: The BERT model to use. One of ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.model = BERT_LM(model)
        self.model_name = model
        self.__name__ = f"{model}_lm_reduce_" + reduce.__name__
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        inp = row.title.lower() if "uncased" in self.model_name else row.title
        return self.model.calc_perplexity_custom_reduce(inp, reduce=self.reduce)


################ Perplexity and AoA ####################


class WordLenAndRarity(object):
    """
    A combination of word length and word rarity classifier. We posit that funny titles should have rare,
    simple words.
    """

    def __init__(
        self,
        counts_path: str = "resources/ngram-language-models/science_unigrams.p",
        reduce=np.mean,
    ):
        """
        :param counts_path: Path to a counter of words, trained on our titles dataset.
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.counts = pickle.load(open(counts_path, "rb"))
        self.total_counts = sum(self.counts.values())
        self.__name__ = "word_len_and_rarity_" + reduce.__name__
        self.reduce = reduce

    def _get_word_prob(self, word: str) -> float:
        word_count = max(self.counts[word], 1)
        return word_count / self.total_counts

    def __call__(self, row: pd.Series) -> float:
        words_rarities = np.array(
            [np.log2(self._get_word_prob(t.lemma_.lower())) for t in row.nlp]
        )
        words_lens = np.array([len(t.text) for t in row.nlp])
        return float(self.reduce(words_rarities + words_lens))


class NgramEntropyAndAoA(object):
    """
    A combination of AoA and Perplexity classifier. We assume that funny titles should have surprising,
    simple words.
    """

    def __init__(
        self,
        type: str = "science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        """
        :param type: The type of the model. can be either "science" or "jokes".
        :param grams: one of  the values in [1,2,3]
        :param pos_tags: Whether to use a pos tags based N-gram model or not
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"AoA_and_{type}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(type, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        probs = np.array(
            [np.log2(self.lm_model[gram]) for gram in title_as_ngrams]
        )  # perplexity

        words_aoas = np.array([aoa_model[t.lemma_.lower()] for t in row.nlp])
        words_aoas = words_aoas[self.ngrams - 1 :]  # AoA per word
        try:
            return float(
                self.reduce(probs / words_aoas)
            )  # combination of AoA and perplexity such that
            # simple, surprising words have higher values than complex, common words
        except ValueError:
            return 0.0


class NgramEntropyAndFunniness(object):
    """
    A combination of word funniness and Perplexity classifier. We assume that funny titles should have surprising,
    funny words.
    """

    def __init__(
        self,
        type: str = "science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        """
        :param type: The type of the model. can be either "science" or "jokes".
        :param grams: one of  the values in [1,2,3]
        :param pos_tags: Whether to use a pos tags based N-gram model or not
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
         """
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"funniness_and_{type}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(type, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        probs = np.array(
            [np.log2(self.lm_model[gram]) for gram in title_as_ngrams]
        )  # perplexity

        words_funs = np.array([fun_model[t.lemma_.lower()] for t in row.nlp])
        words_funs = words_funs[self.ngrams - 1 :]  # funniness of each word
        try:
            return float(
                self.reduce(probs * words_funs)
            )  # combination of funniness and perplexity such that
            # funny, surprising words have higher values than serious, common words
        except ValueError:
            return 0


################ syntax based functions ####################


class PosTagsStatistics(object):
    """
    Simple classifier that counts how many words of a specific pos tag
    the title contains
    """

    types = ["ADJ", "NOUN"]

    def __init__(self, pos_tag: str):
        """
        :param pos_tag: One of the pos tags. We discovered that only ["ADJ", "NOUN"] are significant
        """
        self.pos_ = pos_tag
        self.__name__ = f"POS_tags_analysis_{pos_tag}"

    def __call__(self, row: pd.Series) -> float:
        counts = [1 for t in row.nlp if t.pos_ == self.pos_]
        return sum(counts) / len(row.nlp)


################ sex and more ####################


def simple_funniness_analysis(row: pd.Series, reduce=np.min) -> float:
    """
    Analyze the sentence using a word funniness model.
    :param row: Row of a titles dataframe
    :param reduce: How to aggregate the results of a token-level statistics
    (default is taking the mean across the sentence, other options are min, max and std)
    """
    mean_aoa = float(reduce([fun_model[toc.lemma_.lower()] for toc in row["nlp"]]))
    return mean_aoa


class RudenessClassifier(object):
    """
    Calculates how likely it is for the title to contain crude or rude language
    """

    def __init__(self):
        self.__name__ = "rudeness_classifier"
        self.model = pickle.load(
            open("resources/rudeness-classifier/rudeness_classifier.m", "rb")
        )

    def __call__(self, row: pd.Series) -> float:
        probs = self.model.predict_proba([row.title])[0]
        return float(np.log2(probs[1]))


class RudenessAndPerplexity(object):
    """
    A combination of rudeness and Perplexity classifier. We assume that funny titles should have surprising,
    rude words.
    """

    def __init__(
        self,
        type: str = "science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        """
        :param type: The type of the model. can be either "science" or "jokes".
        :param grams: one of  the values in [1,2,3]
        :param pos_tags: Whether to use a pos tags based N-gram model or not
        :param reduce: How to aggregate the results of a token-level statistics
        (default is taking the mean across the sentence, other options are min, max and std)
        """
        self.model = pickle.load(
            open("resources/rudeness-classifier/rudeness_classifier.m", "rb")
        )
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"_{type}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ = "rudeness_classifier" + self.__name__ + "_" + reduce.__name__
        self.lm_model = NgramBasedLM(type, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        perplexity_probs = np.array(
            [np.log2(self.lm_model[gram]) for gram in title_as_ngrams]
        )  # perplexity

        rudeness_probs = [
            np.log2(self.model.predict_proba([t.text.lower()])[0][1]) for t in row.nlp
        ]
        rudeness_probs = [min(item, -1) for item in rudeness_probs]
        rudeness_probs = np.array(rudeness_probs[self.ngrams - 1 :])

        try:
            return float(
                self.reduce(perplexity_probs / rudeness_probs)
            )  # combination of rudeness and perplexity
            # such that rude, surprising words have higher values than benign, common words
        except ValueError:
            return 0.0
