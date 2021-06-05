from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import pickle

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import (
    BertPreTrainedModel,
    BertModel,
    BertForSequenceClassification,
)
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from run_classifier_dataset_utils_refactored import (
    IggProcessor,
    convert_examples_to_features,
    compute_metrics,
    NUM_FEATURES,
    FIELDS,
)


logger = logging.getLogger(__name__)
SCI_BERT = r"/cs/labs/dshahaf/d8200/downloads/scibert_scivocab_uncased"
SCI_BERT_TOKENIZER = r"/cs/labs/dshahaf/d8200/downloads/scibert_scivocab_uncased"
BIO_BERT = ""  # TODO add


class FeatureNet(BertPreTrainedModel):  # TODO see how to freeze the weights
    def __init__(self, config, num_labels, features=False):
        super(FeatureNet, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if features:
            self.classifier = nn.Linear(config.hidden_size + NUM_FEATURES, num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        features=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
    ):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False
        )
        pooled_output = self.dropout(pooled_output)
        if features is not None:
            pooled_output = torch.cat((pooled_output, features), 1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--save_examples",
        action="store_true",
        help="Whether to save the processed train and test examples",
    )
    parser.add_argument(
        "--weights_dir",
        default="",
        type=str,
        help="Path to the weights of a bert model.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Whether to use field specific features.",
    )
    parser.add_argument(
        "--filter_train",
        action="store_true",
        help="Whether to filter the train examples using estimator model.",
    )
    parser.add_argument(
        "--filter_test",
        action="store_true",
        help="Whether to filter the test examples using estimator model.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_seq_length", default=128, type=int, help="maximal sentence length"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    args = parser.parse_args()
    return args


def main(args):

    ################################################
    # setting the ground for training
    ################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # set random seeds to some number
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(42)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = IggProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    ################################################ +:)
    # here we load the model
    ################################################ +:)

    if args.weights_dir:
        model_name = args.weights_dir
    elif args.bert_model == "sci-bert":
        model_name = SCI_BERT
    else:
        model_name = args.bert_model

    tokenizer = BertTokenizer.from_pretrained(SCI_BERT, do_lower_case=True)
    # model = FeatureNet.from_pretrained(model_name, num_labels=num_labels, features=args.use_features)
    model = BertForSequenceClassification.from_pretrained(
        SCI_BERT, num_labels=num_labels
    )
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss = 0

    # if we want to train
    if args.do_train:
        tb_writer = SummaryWriter()

        ################################################ +:)
        # We start with loading and preprocessing the examples
        ################################################ +:)

        # search for saved examples, or create and dump them
        train_examples = processor.get_train_examples(args.data_dir)
        cached_train_features_file = os.path.join(
            args.data_dir,
            "train_{0}_{1}_{2}".format(
                args.bert_model,
                str(args.max_seq_length),
                "use_features: " + str(args.use_features),
            ),
        )
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer
            )
            logger.info(
                "  Saving train features into cached file %s",
                cached_train_features_file,
            )
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)

        # convert the data to tensors
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.long
        )

        if args.use_features:
            all_features = torch.tensor(
                [f.features for f in train_features], dtype=torch.float
            )
            train_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_features,
                all_label_ids,
            )
        else:
            train_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )

        n_train_steps = len(train_dataloader) * args.num_train_epochs

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            warmup=0.1,
            t_total=n_train_steps,
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", n_train_steps)

        ################################################ +:)
        # the actual train
        ################################################ +:)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                if args.use_features:
                    input_ids, input_mask, segment_ids, features, label_ids = batch
                    logits = model(
                        input_ids,
                        features,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                    )
                else:
                    input_ids, input_mask, segment_ids, label_ids = batch
                    logits = model(
                        input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                    )

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                tb_writer.add_scalar("lr", optimizer.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss.item(), global_step)

        # Save a trained model, configuration and tokenizer
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)
        output_args_file = os.path.join(args.output_dir, "training_args.bin")
        torch.save(args, output_args_file)

    ################################################ +:)
    # evaluating the model
    ################################################ +:)

    if args.do_eval:

        ################################################ +:)
        # loading the model if we need to
        ################################################ +:)

        if not args.do_train:
            model_name = args.weights_dir[args.weights_dir.rfind(r"/") + 1 :]

            tokenizer = BertTokenizer.from_pretrained(
                args.weights_dir, do_lower_case=True
            )
            model = BertForSequenceClassification.from_pretrained(
                args.weights_dir, num_labels=num_labels
            )
            model.to(device)

        ################################################ +:)
        # loading and preparing the dataset
        ################################################ +:)

        eval_examples = processor.get_dev_examples(args.data_dir)
        cached_eval_features_file = os.path.join(
            args.data_dir,
            "dev_{0}_{1}_{2}".format(
                args.bert_model,
                str(args.max_seq_length),
                "use_features: " + str(args.use_features),
            ),
        )
        try:
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer
            )
            logger.info(
                "  Saving eval features into cached file %s", cached_eval_features_file
            )
            with open(cached_eval_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long
        )
        if args.use_features:
            all_features = torch.tensor(
                [f.features for f in eval_features], dtype=torch.float
            )
            eval_data = TensorDataset(
                all_input_ids,
                all_input_mask,
                all_segment_ids,
                all_features,
                all_label_ids,
            )
        else:
            eval_data = TensorDataset(
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids
            )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.train_batch_size
        )

        ################################################ +:)
        # the actual evaluation
        ################################################ +:)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None
        all_features = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                if not args.use_features:
                    input_ids, input_mask, segment_ids, label_ids = batch
                    logits = model(
                        input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                    )
                else:
                    input_ids, input_mask, segment_ids, features, label_ids = batch
                    logits = model(
                        input_ids,
                        features,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                    )

            # create eval loss and other metric required by the task
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
                if args.use_features:
                    all_features = features.detach().cpu().numpy()
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                )
                if args.use_features:
                    all_features = np.append(
                        all_features, features.detach().cpu().numpy(), axis=0
                    )

        if args.use_features:
            fields = all_features[:, -10:]
            fields = np.argmax(fields, axis=1)
        else:
            fields = None

        ################################################ +:)
        # Computing the loss
        ################################################ +:)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        loss = tr_loss / global_step if args.do_train else None

        result["eval_loss"] = eval_loss
        result["global_step"] = global_step
        result["train loss"] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        precision, recall, f_score, _ = precision_recall_fscore_support(
            out_label_ids, preds, average="binary"
        )
        accuracy = accuracy_score(out_label_ids, preds)
        print(
            f"Scores for SciBert classifier on test:\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 score:\t{f_score}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
