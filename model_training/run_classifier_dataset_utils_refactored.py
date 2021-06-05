# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import textstat
from textblob import TextBlob
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MAX_LEN = 400


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self, guid, text_a, text_b=None, field=None, label=None, features=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.field = field
        self.label = label
        self.features = features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, features=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.features = features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:  # , encoding="ISO-8859–1") as f:
            reader = csv.reader(f)  # , delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class IggProcessor(DataProcessor):
    """Processor for the Igg data set."""

    def get_train_examples(self, data_dir, features=False):
        """See base class."""
        if features:
            return self._create_examples_with_features(
                os.path.join(data_dir, "train.csv")
            )[0]
        else:
            return self._create_examples(os.path.join(data_dir, "train.csv"))[0]

    def get_dev_examples(self, data_dir, features=False):
        """See base class."""
        if features:
            return self._create_examples_with_features(
                os.path.join(data_dir, "test.csv")
            )[0]
        else:
            return self._create_examples(os.path.join(data_dir, "test.csv"))[0]

    def get_predict_examples(self, data_dir, features=False):
        """See base class."""
        if features:
            return self._create_examples_with_features(
                os.path.join(data_dir, "predict.csv"), predict=True
            )
        else:
            return self._create_examples(
                os.path.join(data_dir, "predict.csv"), predict=True
            )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, dataset_path, predict=False):
        """Creates examples for the training and dev sets."""
        examples = []
        data = pd.read_csv(dataset_path)
        if not predict:
            for i, row in data.iterrows():
                guid = "%s" % (row.id)
                text_a = row.title
                field = row.field
                label = row.label
                examples.append(
                    InputExample(
                        guid=guid, text_a=text_a, text_b=None, field=field, label=label
                    )
                )
        else:
            for i, row in data.iterrows():
                guid = "%s" % (row.id)
                text_a = row.title
                examples.append(
                    InputExample(
                        guid=guid, text_a=text_a, text_b=None, field=None, label=0
                    )
                )

        return examples, data

    def _get_feature_names(self, dataset):
        names = [name for name in list(dataset.columns) if "result" in name]
        names = list(
            filter(lambda x: "gpt2_lm_after_fine_tuning_reduce_" not in x, names)
        )
        assert len(names) == 127
        return names

    def _create_examples_with_features(self, dataset_path, predict=False):
        """Creates examples for the training and dev sets."""
        examples = []
        data = pd.read_csv(dataset_path)
        feature_names = self._get_feature_names(data)
        if not predict:
            for i, row in tqdm(data.iterrows()):
                guid = "%s" % (row.id)
                text_a = row.title
                field = row.field
                label = row.label
                features = np.array(row[feature_names])
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=None,
                        field=field,
                        label=label,
                        features=features,
                    )
                )
        else:
            for i, row in tqdm(data.iterrows()):
                guid = "%s" % (row.id)
                text_a = row.title
                features = np.array(row[feature_names])
                examples.append(
                    InputExample(
                        guid=guid,
                        text_a=text_a,
                        text_b=None,
                        field=None,
                        label=0,
                        features=features,
                    )
                )

        return examples, data


def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, use_features=False
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if use_features:
            my_features = example.features
        else:
            my_features = None

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[str(example.label)]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                features=my_features,
            )
        )
    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def acc_recall_presicion(preds, labels):
    res = {
        "count": labels.shape[0],
        "acc": simple_accuracy(labels, preds),
        "presicion": precision_score(y_true=labels, y_pred=preds),
        "recall": recall_score(y_true=labels, y_pred=preds),
    }
    return res


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_recall_presicion(preds, labels)
