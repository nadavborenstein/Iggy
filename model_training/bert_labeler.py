import argparse
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from tqdm import tqdm
from run_classifier_dataset_utils_refactored import (
    IggProcessor,
    convert_examples_to_features,
    compute_metrics,
)
from bert_classifier_training_with_features import SimpleFeatureNet


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
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length", default=128, type=int, help="maximal sentence length"
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--use_features", action="store_true", help="Whether to use features or not"
    )
    args = parser.parse_args()
    return args


def get_features(args):
    processor = IggProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    predict_examples, dataset = processor.get_predict_examples(
        args.data_dir, features=args.use_features
    )
    label_list = processor.get_labels()
    predict_features = convert_examples_to_features(
        predict_examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        use_features=args.use_features,
    )
    return predict_features, dataset


def predict(args, features):
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 2
    args.device = device
    if args.use_features:
        model = SimpleFeatureNet.from_pretrained(
            args.bert_model,
            num_labels=num_labels,
            features_hidden_size1=512,
            features_hidden_size2=512,
            features_dropout=0.5,
            classifier_hidden_size=1024,
        )
    else:
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model, num_labels=2
        )
    model.to(device)
    model.eval()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.use_features:
        all_feature = torch.tensor([f.features for f in features], dtype=torch.float)
        data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_feature
        )
    else:
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            if args.use_features:
                input_ids, input_mask, segment_ids, features = batch
                logits = model(
                    input_ids,
                    features,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                )
            else:
                input_ids, input_mask, segment_ids = batch
                logits = model(
                    input_ids, token_type_ids=segment_ids, attention_mask=input_mask
                )

        if len(preds) == 0:
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], torch.sigmoid(logits).detach().cpu().numpy(), axis=0
            )

    preds = preds[0]
    return preds[:, 1]


def main():
    args = parse_args()
    features, dataset = get_features(args)
    predictions = predict(args, features)
    dataset["prediction"] = predictions
    dataset.to_csv(args.output_path)


if __name__ == "__main__":
    main()
