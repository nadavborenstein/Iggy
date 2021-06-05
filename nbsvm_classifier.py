# Copyright (c) 2019 Lightricks. All rights reserved.
import re
import string
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_is_fitted


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    """
    Implementation of a simple Naive Bayes SVM classifier for textual
    input (first presented in https://www.aclweb.org/anthology/P12-2018/).
    """

    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            tokenizer=self._tokenize,
            min_df=3,
            max_df=0.9,
            strip_accents="unicode",
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1,
        )

    def _tokenize(self, s):
        re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
        return re_tok.sub(r" \1 ", s).split()

    def predict(self, x):
        # Verify that model has been fit
        x = self.vec.transform(x)
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        x = self.vec.transform(x)
        check_is_fitted(self, ["_r", "_clf"])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x = self.vec.fit_transform(x)
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(
            C=self.C, dual=self.dual, n_jobs=self.n_jobs
        ).fit(x_nb, y)
        return self
