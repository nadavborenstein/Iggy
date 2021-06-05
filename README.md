# Iggy
[[Paper]]() [[Presentation]]()

Implementation of the paper "How Did This Get Funded?!Automatically Identifying Quirky Scientific Achievements"


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



## Usage

### Using the classifiers

To analyze a set of title using the classifiers, `cd` to `Iggy/` and run:

```bash
python run_classifiers.py --titles_path TITLES_PATH --save_path SAVE_PATH
                          --classifiers_to_use CLASSIFIERS_TO_USE [--labeled]
```

Where:

`--titles_path`: path to a csv of titles to analyze. The titles should be in a column named "title".

`save_path`: where to save the analysis results.

`classifiers_to_use`: which classifiers to use to analyze the titles. A path to a txt file with the names of the classifiers to run.
Check `classifiers/classifiers_names.txt` for reference.

`--labeled`: whether the titles to analyze are labeled. That is, whether the titles come from the file
`datasets/IgDataset.csv`


After the code will finish running, you should see in SAVE_PATH a file with the same structure as `dataset/IggDataset_with_data_analysis_results.csv`

### Training the MLP model

### Training the BERT-based models

### 