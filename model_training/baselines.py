import pandas as pd
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from os.path import join


def load_data(root_path: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Loads the training ant testing datasets
    """
    train = pd.read_csv(join(root_path, "/train.csv")).sample(frac=1.0, random_state=42)
    test = pd.concat(
        [
            pd.read_csv(join(root_path, "/test.csv")),
            pd.read_csv(join(root_path, "/dev.csv")),
        ],
        0,
    )  # no need for dev

    X_train, y_train = train.title, train.label
    X_test, y_test = test.title, test.label
    return X_train, y_train, X_test, y_test


def define_model(model_type: str) -> Pipeline:
    """
    Defines a simple baseline model based on bag-of-words.
    """
    bow = CountVectorizer(
        strip_accents="ascii",
        lowercase=True,
        stop_words="english",
        max_df=1.0,
        min_df=3,
        max_features=None,
        binary=True,
    )
    if model_type == "RF":
        model = RandomForestClassifier(random_state=42, verbose=1)
    elif model_type == "LR":
        model = LogisticRegression(verbose=1)
    pipeline = Pipeline([("BoW", bow), ("model", model)])
    return pipeline


def train_model(model: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Trains the model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """
    Evaluate the model.
    """
    print("score:", model.score(X_test, y_test))


if __name__ == "__main__":
    model_types = ["RF", "LR"]
    root_path = "datasets/train-data/"

    X_train, y_train, X_test, y_test = load_data(root_path)
    for model_type in model_types:
        model = define_model(model_type)
        model = train_model(model, X_train, y_train)
        evaluate_model(model, X_test, y_test)
