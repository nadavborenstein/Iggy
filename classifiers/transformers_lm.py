import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM
import numpy as np


class GPT2LM(object):
    """
    A language model based on GPT-2.
    """

    def __init__(self, model_name_or_path: str = "gpt2"):
        """
        Init the language model.
        :param model_name_or_path: Path to saved checkpoint, or to a model name
        (one of the models supported by https://huggingface.co/). Default is gpt2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    def _tokenize(self, sentence: str) -> (torch.Tensor, list):
        """
        Tokenize a sentence so it could be fed to the model
        """
        indexed_tokens = self.tokenizer.encode(sentence)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(self.device)
        return tokens_tensor, indexed_tokens

    def calc_perplexity(self, sentence: str) -> float:
        """
        Calculates the perplexity of a sentence by returning the next-word prediction loss of GPT-2
        """
        sentence_tensor, _ = self._tokenize(sentence)
        with torch.no_grad():
            loss, _, _ = self.model(sentence_tensor, labels=sentence_tensor)
        return float(loss)

    def calc_perplexity_custom_reduce(self, sent: str, reduce=np.mean) -> float:
        """
        Compute sentence "perplexity" while using a custom reduction (this is the same as regular
        perplexity if reduce=mean).
        :param reduce: the custom reduce to use. Default is np.mean.
        """
        sentence_tensor, indexed_tokens = self._tokenize(sent)
        context = sentence_tensor[:, :1]
        past = None
        perplexities = []
        for i in range(1, sentence_tensor.shape[1]):
            output, past = self.model(context, past=past)
            after_softmax = torch.softmax(output, 2)
            perplexities.append(
                np.log2(after_softmax[..., -1, indexed_tokens[i]].to("cpu").item())
            )
            context = sentence_tensor[:, i : i + 1]

        return float(reduce(perplexities))


class BERT_LM(object):
    """
    A language model based on BERT
    """

    def __init__(
        self, model_name_or_path: str = "bert-base-uncased",
    ):
        """
        Init the language model
        :param model_name_or_path:  Path to saved checkpoint, or to a model name
        (one of the models supported by https://huggingface.co/). Default is bert-base-uncased.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.model.eval()
        self.model.to(self.device)

    def _tokenize(self, sentence: str) -> (torch.Tensor, list):
        """
        Tokenize a sentence so it could be fed to the model
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        return inputs.to(self.device)

    def _tokenize_with_masks(self, sentence: str) -> (torch.Tensor, list):
        """
        Tokenize a sentence for perplexity calculation. For each word in the sentence,
        create an input in which this specific word is masked. The batch consists of the concatenation of
        all those inputs
        """
        tokenized_text = self.tokenizer.tokenize(sentence)
        inputs_with_mask = []
        for i in range(len(tokenized_text)):
            temp = tokenized_text.copy()
            temp[i] = "[MASK]"
            temp = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + temp + ["[SEP]"])
            inputs_with_mask.append(temp)
        inputs_with_mask = torch.tensor(inputs_with_mask)
        inputs_with_mask = inputs_with_mask.to(self.device)
        return inputs_with_mask

    def calc_entire_sentence_perplexity(self, sentence: str) -> float:
        """
        Calculates the perplexity of a sentence by using the loss of the model
        """
        tokenized_sentence = self._tokenize(sentence)
        with torch.no_grad():
            outputs = self.model(
                **tokenized_sentence.data, labels=tokenized_sentence["input_ids"]
            )
        loss, prediction_scores = outputs[:2]
        return loss.item()

    def calc_perplexity_custom_reduce(self, sentence: str, reduce=np.mean) -> float:
        """
        Calculates the perplexity of a sentence by using a Masked Language Model loss and custom reduce.
        :param reduce: the custom reduce to use. Default is np.mean.
        """
        tokenized_sentence = self._tokenize(sentence)
        tokenized_sentence_with_masks = self._tokenize_with_masks(sentence)
        with torch.no_grad():
            outputs = self.model(tokenized_sentence_with_masks)[0]
            after_softmax = torch.softmax(outputs, 2)
        probs = []
        for i in range(1, outputs.shape[1] - 1):
            prob = after_softmax[i - 1, i, tokenized_sentence["input_ids"][0][i]]
            probs.append(np.log2(prob.to("cpu").item()))
        return float(reduce(probs))
