import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
import pickle
import os


def parse_args():
    """
    Parse command line arguments.
    :return:
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_save_path",
        default=None,
        type=str,
        required=True,
        help="where to store the model.",
    )
    parser.add_argument(
        "--train", action="store_true", help="whether to train the model",
    )
    parser.add_argument(
        "--test", action="store_true", help="whether to test the model",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="whether to use the model for prediction",
    )

    parser.add_argument(
        "--train_dataset_root_path",
        default="dataset/",
        type=str,
        required=False,
        help="root path of the dataset",
    )
    parser.add_argument(
        "--predict_test_dataset_path",
        type=str,
        required=False,
        help="path of the dataset to predict or test on",
    )
    parser.add_argument(
        "--predict_output_path",
        type=str,
        required=False,
        help="path of the prediction results",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        required=False,
        help="hidden size of the MLP model.",
    )
    parser.add_argument(
        "--alpha",
        default=2.0,
        type=float,
        required=False,
        help="l2 regularization parameter.",
    )
    args = parser.parse_args()
    return args


def load_data(path: str) -> pd.DataFrame:
    """
    Loads a csv file with pandas
    """
    if path is None:
        return None
    else:
        return pd.read_csv(path)


def load_datasets(
    train_path: str, dev_path: str, test_path: str
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Loads the train, dev and test set
    """
    return load_data(train_path), load_data(dev_path), load_data(test_path)


def get_feature_columns_names(dataset: pd.DataFrame) -> [str]:
    """
    Get the names of all the features the MLP model should use
    """
    names = [name for name in list(dataset.columns) if "result" in name]
    return names


def nan_rm(features: pd.DataFrame, names) -> pd.DataFrame:
    """
    Removes NaNs from the data
    """
    means = features[names].mean()
    fillnans = features[names].fillna(means)
    features[names] = fillnans
    return features


def norm_features(features: pd.DataFrame) -> (StandardScaler, pd.DataFrame):
    """
    Scale the features and return the normalized features and the scaler
    """
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def data_prepare(root_path: str, scale: bool = True):
    """
    Preprocess the data
    :param scale: whether to use a standard scaler
    """
    train_path = root_path + "/train-data/train.csv"
    dev_path = root_path + "/train-data/dev.csv"
    test_path = root_path + "/train-data/test.csv"

    train, dev, test = load_datasets(train_path, dev_path, test_path)
    feature_names = get_feature_columns_names(train)

    if scale:
        scaler = norm_features(train[feature_names])
        train[feature_names] = scaler.transform(train[feature_names])
        dev[feature_names] = scaler.transform(dev[feature_names])
        test[feature_names] = scaler.transform(test[feature_names])
        pickle.dump(scaler, open("models-weights/scalerMLP.sav", "wb"))

    return train, dev, test


def model_training(X_train, X_dev, y_train, y_dev, args):
    """
    Train the MLP classifier
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(args.hidden_size,), alpha=args.alpha, shuffle=False
    )
    mlp.fit(X_train, y_train)
    print("Train score is:", mlp.score(X_train, y_train))
    print("Dev. score is:", mlp.score(X_dev, y_dev))
    return mlp


def train(args):
    """
    Main function for training the MLP model
    """
    train, dev, _ = data_prepare(root_path=args.train_dataset_root_path, scale=True)
    feature_names = get_feature_columns_names(train)

    X_train = train[feature_names]
    X_dev = dev[feature_names]

    y_train = train.label
    y_dev = dev.label
    mlp = model_training(X_train, X_dev, y_train, y_dev, args)
    pickle.dump(mlp, open(args.model_save_path, "wb"))


def test(args):
    print("loading data")
    data = pd.read_csv(args.predict_test_dataset_path)
    names = get_feature_columns_names(data)

    print("remove nans")
    data = nan_rm(data, names)

    print("scaling")
    scaler = pickle.load(open("models-weights/scalerMLP.sav", "rb"))
    model = pickle.load(open(args.model_save_path, "rb"))
    data[names] = scaler.transform(data[names])
    X = data[names]

    print("testing")
    predictions = model.predict(X)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        data.label, predictions, average="binary"
    )
    accuracy = accuracy_score(data.label, predictions)
    print(
        f"Scores for MLP classifier on test:\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 score:\t{f_score}"
    )


def predict(args):
    """
    Main function of prediction
    """
    print("loading data")
    data = load_data(args.predict_test_dataset_path)
    names = get_feature_columns_names(data)

    print("remove nans")
    data = nan_rm(data, names)
    scaler = pickle.load(open("models-weights/scalerMLP.sav", "rb"))
    model = pickle.load(open(args.model_save_path, "rb"))

    print("scaling")
    data[names] = scaler.transform(data[names])
    X = data[names]

    print("predicting")
    predictions = model.predict_proba(X)
    data["mlp_predictions"] = predictions[:, 1]

    print("saving")
    data.to_csv(
        args.predict_output_path, index=False,
    )


def main():
    args = parse_args()
    if args.train:
        train(args)

    if args.test:
        test(args)
    if args.predict:
        predict(args)


if __name__ == "__main__":
    main()
