import argparse

import matplotlib.pyplot as plt
import seaborn as sb
import spacy
from scipy import stats
from tqdm import tqdm
import pandas as pd
from typing import Callable
from classifiers import get_function_from_name

################ consts #####################

FILEDS = ["neuroscience", "exact_eng", "bio_env", "medicine"]
nlp = spacy.load("en_core_web_sm")
tqdm.pandas()


################ utils #####################


def t_test(pop_a: pd.Series, pop_b: pd.Series, f: Callable) -> float:
    """
    Perform a simple t-test
    :param pop_a: Population 1 of titles, a pd.Series with the relevant classifier results
    :param pop_b: Population 2 of titles
    :param f: Classifier to calculate t-score with
    """
    pop_a = pop_a[f"{f.__name__}_result"].values
    pop_b = pop_b[f"{f.__name__}_result"].values
    statistics, p_value = stats.ttest_ind(pop_a, pop_b, equal_var=False)
    return p_value


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--titles_path",
        default=None,
        type=str,
        required=True,
        help="path to a csv of titles to analyze.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=str,
        required=True,
        help="where to save the analysis results.",
    )
    parser.add_argument(
        "--classifiers_to_use",
        default=None,
        type=str,
        required=True,
        help="path to a txt file with the names of the classifiers to run.",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="whether the titles to analyze are labeled.",
    )
    args = parser.parse_args()
    return args


def spacy_analysis(data: pd.DataFrame) -> pd.DataFrame:
    data["nlp"] = data["title"].progress_apply(nlp)
    return data


################ main #####################


def analyze_dataset_using_function(
    data: pd.DataFrame, f: Callable, plot: bool = False, labeled: bool = False
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
    titles = pd.read_csv(args.titles_path).sample(100)
    titles = spacy_analysis(titles)

    classifiers_to_use = [clf[:-1] for clf in open(args.classifiers_to_use).readlines()]
    for i, classifier_name in enumerate(classifiers_to_use):
        classifier = get_function_from_name(classifier_name)

        print("Now analyzing", classifier.__name__)
        titles, a, b = analyze_dataset_using_function(
            titles, classifier, plot=False, labeled=args.labeled
        )
        res = t_test(a, b, classifier)
        print(f"t-test score for {classifier.__name__} is {res}")

    print("Done")
    titles.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
