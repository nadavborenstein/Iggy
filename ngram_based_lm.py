import pickle
import numpy as np
import pandas as pd
import spacy

UNIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/unigrams/{}_unigrams.p"
BIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/bigrams/{}_bigrams.p"
TRIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/trigrams/{}_trigrams.p"

POS_UNIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/unigrams/{}_unigrams_pos_tags.p"
POS_BIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/bigrams/{}_bigrams_pos_tags.p"
POS_TRIGRAMS_PATH = "/Users/nborenstein/PycharmProjects/Iggy_AAAI/data/trigrams/{}_trigrams_pos_tags.p"




class NgramBasedLM(object):

    def __init__(self, field: str, grams: int = 2, pos_tags: bool = False):
        self.field = field
        self.grams = grams
        if pos_tags:
            self.unigrams = pickle.load(open(POS_UNIGRAMS_PATH.format(field), "rb"))
            self.bigrams = pickle.load(open(POS_BIGRAMS_PATH.format(field), "rb"))
            self.trigrams = pickle.load(open(POS_TRIGRAMS_PATH.format(field), "rb")) if grams == 3 else None
        else:
            self.unigrams = pickle.load(open(UNIGRAMS_PATH.format(field), "rb"))
            self.bigrams = pickle.load(open(BIGRAMS_PATH.format(field), "rb"))
            self.trigrams = pickle.load(open(TRIGRAMS_PATH.format(field), "rb")) if grams == 3 else None
        self.unigrams_total = sum(self.unigrams.values())

    def __getitem__(self, ngram: tuple) -> float:
        if self.grams == 3:
            w_2, w_1, w = ngram
            trigram_count = self.trigrams[(w_2, w_1, w)]
            w_1_w_2_bigram_count = self.bigrams[(w_2, w_1)]
            bigrams_count = self.bigrams[(w_1, w)]
            w_1_unigram_count = self.unigrams[w_1]
        elif self.grams == 2:
            w_1, w = ngram
            bigrams_count = self.bigrams[(w_1, w)]
            w_1_unigram_count = self.unigrams[w_1]
            trigram_count, w_1_w_2_bigram_count = 0, 0
        else:
            w = ngram[0]
            trigram_count, w_1_w_2_bigram_count, bigrams_count, w_1_unigram_count = 0, 0, 0, 0

        w_unigram_count = self.unigrams[w]
        if trigram_count != 0:
            return trigram_count / w_1_w_2_bigram_count
        elif bigrams_count != 0:
            return bigrams_count / w_1_unigram_count
        else:
            return max(w_unigram_count / self.unigrams_total, 1 / self.unigrams_total)

    def _generate_ngrams(self, dataset: pd.DataFrame, pos_tags: bool, save_path: str = "resources/ngram_language_models/"):
        nlp = spacy.load("en_core_web_sm")

    def sentence_entropy(self, sent_as_ngram: list) -> float:
        entropy = np.mean([np.log2(self[ngram]) for ngram in sent_as_ngram])
        return float(entropy)

    def sentence_mean_prob(self, sent_as_ngram: list) -> float:
        mean_prob = np.mean([self[ngram] for ngram in sent_as_ngram])
        return float(mean_prob)

    def sentence_entropy_custom_reduce(self, sent_as_ngram: list, reduce = np.mean) -> float:
        entropy = reduce([np.log2(self[ngram]) for ngram in sent_as_ngram])
        return float(entropy)



###### FUNCTIONS RELATED TO CREATION OF THE PICKLES ######


def n_gram_stats(field, nlp, n, abstract=False):
    """
    Computes the n grams model for a field with/without abstract
    """
    nouns_dict = defaultdict(int)
    data = pd.read_csv(os.path.join(PATH_TO_ARTICLES_FY_FIELD_AFTER_FILTRATION, field + ".csv")) # TODO: change the path once the data changes!
    for index, row in tqdm(data.iterrows()):
        title = row["title"]
        if abstract:
            abstract = row["paperAbstract"]
            doc = nlp(str(title) + " " + str(abstract))
        else:
            doc = nlp(str(title))
        nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" if token.lemma_.isalpha()]

        for i in range(n, len(nouns)):
            window = nouns[i-n:i]
            nouns_dict[tuple(window)] += 1
    return nouns_dict


def normalization_ngrams(field, path, n):
    """
    Normalizes the n grams model and return a logit distribution for each sequance
    """
    # for field in fields:
    with open(path+"{}gram_stats_{}.pickle".format(n, field), 'rb') as f:
        field_dict = pickle.load(f)
        # all_dicts = pickle.load(f)
        # field_dict = all_dicts[field]
        values = field_dict.values()
        counter = sum(values)
        log_counter = np.log(counter)

    for key in field_dict.keys():
        field_dict[key] = np.log(field_dict[key]) - log_counter

    with open(path+"{}gram_stats_{}_normalized.pickle".format(n, field),"wb") as pickle_out:
        pickle.dump(field_dict, pickle_out)
    return counter


def dict_of_dicts_merge(x, y):
    z = x.copy()
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        z[key] = x[key] + y[key]
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z


def inter_fields(fields, path):
    with open(path+"all_fields_freq_dict.p", 'rb') as f:
        all_dicts = pickle.load(f)
        one_two = dict_of_dicts_merge(all_dicts[fields[0]], all_dicts[fields[1]])
        print("1,2")
        three_four =  dict_of_dicts_merge(all_dicts[fields[2]], all_dicts[fields[3]])
        print("3,4")
        five_six = dict_of_dicts_merge(all_dicts[fields[4]], all_dicts[fields[5]])
        print("5,6")
        seven_eight =  dict_of_dicts_merge(all_dicts[fields[6]], all_dicts[fields[7]])
        print("7,8")
        nine_ten = dict_of_dicts_merge(all_dicts[fields[8]], all_dicts[fields[9]])
        print("9,10")

        one_two_three_four =  dict_of_dicts_merge(one_two, three_four)
        print("1-4")
        five_six_seven_eight =  dict_of_dicts_merge(five_six, seven_eight)
        print("5-8")
        one_to_eight = dict_of_dicts_merge(one_two_three_four, five_six_seven_eight)
        print("1-8")
        all = dict_of_dicts_merge(one_to_eight, nine_ten)
        print("all")

        with open(path+"all_fields_freq_dict_normalized.pickle","wb") as pickle_out:
            pickle.dump(all, pickle_out)


def all_fields_stats(n):
    n = str(n)
    nouns_dict = defaultdict(int)
    nlp = spacy.load("en_core_web_sm")
    PATH_TO_ARTICLES_TSV = r"/cs/labs/dshahaf/chenxshani/IgNobel-shared with Nadav/res/articles/" + n + "/"
    for filename in os.listdir(PATH_TO_ARTICLES_TSV):
        data = pd.read_csv(os.path.join(PATH_TO_ARTICLES_TSV, filename), delimiter="\t")
        for index, row in tqdm(data.iterrows()):
            title = row["title"]
            doc = nlp(str(title))
            nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN" if token.lemma_.isalpha()]
            for noun in nouns:
                nouns_dict[noun] += 1

        with open("/cs/labs/dshahaf/chenxshani/IgNobel-shared with Nadav/res/new_ngrams/clusters/1gram_stats_science{}_file_{}.pickle".format(n, filename[:-4]), 'wb') as f:
            pickle.dump(nouns_dict, f)

    with open("/cs/labs/dshahaf/chenxshani/IgNobel-shared with Nadav/res/new_ngrams/clusters/1gram_stats_science{}.pickle".format(n), 'wb') as f:
        pickle.dump(nouns_dict, f)