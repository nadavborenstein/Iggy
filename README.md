# Iggy
[[Paper]](http://arxiv.org/abs/2106.03048)  []([[Presentation]]())

Implementation of the paper "How Did This Get Funded?!Automatically Identifying Quirky Scientific Achievements".


## Downloading relevant files

### The N-gram models: 
Copy all the files from [here](https://drive.google.com/drive/folders/1CuG7WBbvmf9hnHTrJipbUL5gdmy8-xPI?usp=sharing)
and place them in `resources/ngram-language-models/`.

### Finetuned GPT-2 model
We finetuned a GPT-2 model on our dataset of titles. Copy the folder from
[here](https://drive.google.com/drive/folders/1FiqMrM0H76DWzcvVBTdkrreFA0hffrQE?usp=sharing) to `resources/finetuned-gpt2/`.

### Rudeness classifier
We trained a simple nbsvm for detecting rude or crude language, and we use it as one of our classifiers. Copy the model from 
[here](https://drive.google.com/drive/folders/1e687JrzzO_VWLl3cX55wUxicrHVuuuuq?usp=sharing) and paste it in `resources/rudeness-classifier`.

### Our pretrained models
You can find our pretrained models (Iggy, and the BERT based models) [here](). Place them in `models-weights`.


### Our SemanticScholar dataset
Our raw SemanticScholar dataset of 0.6M titles can be found [here](https://drive.google.com/file/d/16AplKQK90o3Eay1QSsbQ7uT5gIZ4gE3I/view?usp=sharing).

It takes some time to run the classifiers on 0.6M titles. Fortunately, you can 
find [here](https://drive.google.com/file/d/1HHnnRgEJcH3Gj0gyTEz6qLp-jVye-tcA/view?usp=sharing) our SemanticScholar dataset 
after performing this step.

## Usage

### Using the classifiers

To analyze a set of title using the classifiers, `cd` to `Iggy/` and run:

```bash
python classifiers/run_classifiers.py --titles_path TITLES_PATH --save_path SAVE_PATH
                          --classifiers_to_use CLASSIFIERS_TO_USE [--labeled]
```

Where:

`titles_path`: path to a csv of titles to analyze. The titles should be in a column named "title".

`save_path`: where to save the analysis results.

`classifiers_to_use`: which classifiers to use to analyze the titles. A path to a txt file with the names of the classifiers to run.
Check `classifiers/classifiers_names.txt` for reference.

`--labeled`: whether the titles to analyze are labeled. That is, whether the titles come from the file
`datasets/IgDataset.csv`


After the code will finish running, you should see in SAVE_PATH a file with the same structure as `dataset/IggDataset_with_data_analysis_results.csv`

### Training, testing and predicting with the MLP model

Usage:
```bash
python model_training/MLP_train.py --model_save_path MODEL_SAVE_PATH [--train] [--test]
                    [--predict]
                    [--train_dataset_root_path TRAIN_DATASET_ROOT_PATH]
                    [--predict_test_dataset_path PREDICT_TEST_DATASET_PATH]
                    [--predict_output_path PREDICT_OUTPUT_PATH]
                    [--hidden_size HIDDEN_SIZE] [--alpha ALPHA]
```

<b>To train the MLP classifier, `cd` to `Iggy/` and run:</b>
```bash
python model_training/MLP_train.py --model_save_path MODEL_SAVE_PATH [--train] 
                    [--train_dataset_root_path TRAIN_DATASET_ROOT_PATH]
                    [--hidden_size HIDDEN_SIZE] [--alpha ALPHA]
```
Where:

`--train`: this will flag the script to train the MLP model.

`model_save_path`: where to save the trained model

`train_dataset_root_path`: path to a directory containing `train.csv` and `dev.csv`. The model will use those files to train.

`hidden_size`: int. Hidden size of the MLP. Default is 256 (this is the value we used).

`alpha`: float. l2 regularization parameter. Default is 2. (this is the value we used).

<br/><br/>
<b>To evaluate the model on labeled data, run:</b>
```bash
python model_training/MLP_train.py --model_save_path MODEL_SAVE_PATH [--test] 
                    [--predict_test_dataset_path PREDICT_TEST_DATASET_PATH]
```

Where:

`--train`: this will flag the script to evaluate the MLP model on `predict_test_dataset_path`.

`model_save_path`: path to the trained model

`predict_test_dataset_path`: path to a labeled dataset to evaluate.

<br/><br/>
<b>To predict using the model on unlabeled data, run:</b>
```bash
python model_training/MLP_train.py --model_save_path MODEL_SAVE_PATH [--predict] 
                    [--predict_test_dataset_path PREDICT_TEST_DATASET_PATH]
                    [--predict_output_path PREDICT_OUTPUT_PATH]
```

Where:

`--predict`: this will flag the script to predict using MLP model on `predict_test_dataset_path`.

`model_save_path`: path to the trained model

`predict_test_dataset_path`: path to an unlabeled dataset to predict on.

`predict_output_path`: where to store the prediction results

### Training and predicting with the BERT-based models

Usage:

To predict using a BERT model on unlabeled data, run:</b>
```bash
 python model_training/bert_labeler.py [-h] --data_dir DATA_DIR --bert_model BERT_MODEL
                       --output_path OUTPUT_PATH
                       [--max_seq_length MAX_SEQ_LENGTH]
                       [--batch_size BATCH_SIZE]
```

### 