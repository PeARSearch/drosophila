import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from drosophila import FFA

categories = ['talk.religion.misc',
              'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
x_train = newsgroups_train.data
y_train = newsgroups_train.target
newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)
x_test = newsgroups_test.data
y_test = newsgroups_test.target


def test_hash_dataset():
    x = np.random.randn(5, 768)
    fly = FFA()
    x_transformed = fly.fit_transform(x)
    assert x_transformed.shape == (5, 100)

def test_vectorizer_pipeline():
    vectorizer = make_pipeline(CountVectorizer(max_features=231, lowercase=False, token_pattern='[^ ]+'),
                               FFA(kc_size=8853, wta=15, proj_size=10))
    x_vec = vectorizer.fit_transform(x_train)
    assert x_vec.shape == (len(x_train), 8853)

def test_classification_pipeline():
    pipeline = make_pipeline(CountVectorizer(max_features=231, lowercase=False, token_pattern='[^ ]+'),
                             FFA(kc_size=8853, wta=15, proj_size=10),
                             LogisticRegression(C=8, max_iter=2000))
    pipeline.fit(x_train, y_train)
    assert pipeline.score(x_train, y_train) == 1.0
