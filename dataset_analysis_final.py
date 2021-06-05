import spacy
import pickle
import argparse
import textstat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from nltk import ngrams
from scipy import stats

from tqdm import tqdm

from AoA_model import AoAModel
from funniness_model import FunModel
from ngram_based_lm import NgramBasedLM
from transformers_lm import GPT2LM, BERT_LM
from nbsvm_classifier import NbSvmClassifier

################ consts #####################

FILEDS = ["neuroscience", "exact_eng", "bio_env", "medicine"]
nlp = spacy.load("en_core_web_sm")
aoa_model = AoAModel()
fun_model = FunModel()

################ utils #####################


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--titles_path",
        default=None,
        type=str,
        required=True,
        help="path to a csv of titles to classify",
    )
    args = parser.parse_args()
    return args


def spacy_analysis(data: pd.DataFrame) -> pd.DataFrame:
    data["nlp"] = data["title"].progress_apply(nlp)
    return data


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
        field: str = "all_science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = field + "_{}grams_entropy".format(grams)
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(field, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp if t.pos_ == "NOUN" ], self.ngrams)
        try:
            entropy = self.lm_model.sentence_entropy_custom_reduce(
                title_as_ngrams, self.reduce
            )
            return float(entropy)
        except ValueError:
            return 0


class SimpleGPT2LM(object):
    def __init__(self, model: str = "models/checkpoint-75000"):
        self.model = GPT2LM(model)
        self.__name__ = "simple_gpt2_lm_after_fine_tuning"

    def __call__(self, row: pd.Series) -> float:
        return self.model.calc_perplexity(row.title.lower())


probs_dict = dict()


class GPT2LMCustomReduce(object):
    def __init__(self, model: str = "models/checkpoint-75000", reduce=np.mean):
        self.model = GPT2LM(model)
        self.__name__ = "gpt2_lm_after_fine_tuning_reduce_" + reduce.__name__
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if row.title in probs_dict:
            probs = probs_dict[row.title]
        else:
            probs = self.model.calc_perplexity_custom_reduce(row.title.lower())
            probs_dict[row.title] = probs
        return float(self.reduce(probs))


class SimpleBERT_LM(object):
    models = ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]

    def __init__(self, model: str = "bert-base-uncased", cache_dir="models/berts"):
        self.model = BERT_LM(model, cache_dir)
        self.model_name = model
        self.__name__ = f"simple_{model}_lm"

    def __call__(self, row: pd.Series) -> float:
        inp = row.title.lower() if "uncased" in self.model_name else row.title
        return self.model.calc_entire_sentence_perplexity(inp)


class BERT_LMCustomReduce(object):
    models = ["bert-base-uncased", "allenai/scibert_scivocab_uncased"]

    def __init__(
        self, model: str = "bert-base-uncased", reduce=np.mean, cache_dir="models/berts"
    ):
        self.model = BERT_LM(model, cache_dir)
        self.model_name = model
        self.__name__ = f"{model}_lm_reduce_" + reduce.__name__
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if row.title in probs_dict:
            probs = probs_dict[row.title]
        else:
            inp = row.title.lower() if "uncased" in self.model_name else row.title
            probs = self.model.calc_perplexity_custom_reduce(inp)[0]
            probs_dict[row.title] = probs
        return float(self.reduce(probs))


################ Perplexity and AoA ####################


class WordLenAndRarity(object):
    def __init__(
        self, counts_path: str = "data/unigrams/all_science_unigrams.p", reduce=np.mean
    ):
        self.counts = pickle.load(open(counts_path, "rb"))
        self.total_counts = sum(self.counts.values())
        self.__name__ = "word_len_and_rarity_" + reduce.__name__
        self.reduce = reduce

    def _get_word_prob(self, word: str):
        word_count = max(self.counts[word], 1)
        return word_count / self.total_counts

    def __call__(self, row: pd.Series) -> float:
        words_rarities = np.array(
            [np.log2(self._get_word_prob(t.lemma_.lower())) for t in row.nlp]
        )
        words_lens = np.array([len(t.text) for t in row.nlp])
        return float(self.reduce(words_rarities + words_lens))


class NgramEntropyAndAoA(object):
    def __init__(
        self,
        field: str = "all_science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"AoA_and_{field}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(field, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        probs = np.array([np.log2(self.lm_model[gram]) for gram in title_as_ngrams])
        words_aoas = np.array([aoa_model[t.lemma_.lower()] for t in row.nlp])
        words_aoas = words_aoas[self.ngrams - 1 :]
        try:
            return float(self.reduce(probs / words_aoas))
        except ValueError:
            return 0.0


class NgramEntropyAndFunniness(object):
    def __init__(
        self,
        field: str = "all_science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"funniness_and_{field}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ += "_" + reduce.__name__
        self.lm_model = NgramBasedLM(field, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        probs = np.array([np.log2(self.lm_model[gram]) for gram in title_as_ngrams])
        words_funs = np.array([fun_model[t.lemma_.lower()] for t in row.nlp])
        words_funs = words_funs[self.ngrams - 1 :]
        try:
            return float(self.reduce(probs * words_funs))
        except ValueError:
            return 0


################ syntax based functions ####################


class PosTagsStatistics(object):
    types = ["ADJ", "NOUN"]

    def __init__(self, pos_tag: str):
        self.pos_ = pos_tag
        self.__name__ = f"POS_tags_analysis_{pos_tag}"

    def __call__(self, row: pd.Series) -> float:
        counts = [1 for t in row.nlp if t.pos_ == self.pos_]
        return sum(counts) / len(row.nlp)


################ sex and more ####################


def simple_funniness_analysis(row: pd.Series, reduce=np.min) -> float:
    mean_aoa = float(reduce([fun_model[toc.lemma_.lower()] for toc in row["nlp"]]))
    return mean_aoa


class RudenessClassifier(object):
    def __init__(self):
        self.__name__ = "rudeness_classifier"
        self.model = pickle.load(open("models/rudeness_classifier.m", "rb"))

    def __call__(self, row: pd.Series) -> float:
        probs = self.model.predict_proba([row.title])[0]
        return float(np.log2(probs[1]))


class RudenessAndPerplexity(object):
    def __init__(
        self,
        field: str = "all_science",
        grams: int = 2,
        pos_tags: bool = False,
        reduce=np.mean,
    ):
        self.model = pickle.load(open("models/rudeness_classifier.m", "rb"))
        self.ngrams = grams
        self.pos_tags = pos_tags
        self.__name__ = f"_{field}_{grams}grams_entropy"
        if pos_tags:
            self.__name__ += "_pos_tags"
        self.__name__ = "rudeness_classifier" + self.__name__ + "_" + reduce.__name__
        self.lm_model = NgramBasedLM(field, grams, pos_tags=pos_tags)
        self.reduce = reduce

    def __call__(self, row: pd.Series) -> float:
        if self.pos_tags:
            tags = [t.tag_ for t in row.nlp]
            title_as_ngrams = ngrams(tags, self.ngrams)
        else:
            title_as_ngrams = ngrams([t.lemma_.lower() for t in row.nlp], self.ngrams)
        perplexity_probs = np.array(
            [np.log2(self.lm_model[gram]) for gram in title_as_ngrams]
        )
        rudeness_probs = [
            np.log2(self.model.predict_proba([t.text.lower()])[0][1]) for t in row.nlp
        ]
        rudeness_probs = [min(item, -1) for item in rudeness_probs]
        rudeness_probs = np.array(rudeness_probs[self.ngrams - 1 :])
        try:
            return float(self.reduce(perplexity_probs / rudeness_probs))
        except ValueError:
            return 0.0


################ main #####################


def analyze_dataset_using_function(
    data: pd.DataFrame, f, plot: bool = False, labeled: bool = False
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    data = data.copy()
    data[f"{f.__name__}_result"] = data.progress_apply(f, axis=1)
    if not labeled:
        return data

    data = data.dropna()
    pos_samples = data[data["ig_winner"] == 1]
    neg_samples = data[(data["label"] == 1) & (data["ig_winner"] == 0)]
    if plot:
        sb.distplot(neg_samples[f"{f.__name__}_result"], bins=30, label="non winner")
        sb.distplot(pos_samples[f"{f.__name__}_result"], bins=30, label="ig winner")
        plt.title(f.__name__.replace("_", " "))
        plt.ylabel("distribution")
        plt.xlabel("value")
        plt.legend()
        plt.show()
    return data, pos_samples, neg_samples


def main():
    args = parse_args()
    titles = pd.read_csv(args.titles_path)
    tqdm.pandas()


if __name__ == "__main__":

    def t_test(pop_a: pd.Series, pop_b: pd.Series, f) -> float:
        pop_a = pop_a[f"{f.__name__}_result"].values
        pop_b = pop_b[f"{f.__name__}_result"].values
        statistics, p_value = stats.ttest_ind(pop_a, pop_b, equal_var=False)
        return p_value

    data = spacy_analysis(data)
    print(data.shape)
    # save_name = sys.argv[1]

    fs = [
        simple_aoa_analysis,
        average_word_len,
        len_analysis,
        Readability(Readability.type),
    ]
    for i, f in enumerate(fs):
        print("Now analyzing", f.__name__)
        data, a, b = analyze_dataset_using_function(data, f, plot=False, labeled=True)
        res = t_test(a, b, f)
        print("for", f.__name__, res)
    print(data.shape)
    # df = pd.DataFrame(results)
