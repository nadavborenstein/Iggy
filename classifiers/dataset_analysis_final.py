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
from classifiers.classifiers import *

################ consts #####################

FILEDS = ["neuroscience", "exact_eng", "bio_env", "medicine"]
nlp = spacy.load("en_core_web_sm")

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
