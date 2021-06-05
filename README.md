# Iggy
[[Paper]]() [[Presentation]]()

Implementation of the paper "How Did This Get Funded?!Automatically Identifying Quirky Scientific Achievements"


## Downloading relevant files
Everything: https://drive.google.com/drive/folders/1Lh9zqrX4m34MBCFIhjw-0jZ_ywJjfMI7?usp=sharing

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

### Training the MLP model

### Training the BERT-based models

### 