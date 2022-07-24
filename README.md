# drosophila
A scikit-learn compatible estimator for the [Fruit Fly Algorithm](https://pearsproject.org/blog/An-introduction-to-the-fruit-fly-algorithm.html) from the [PeARS](https://github.com/PeARSearch/PeARS-fruit-fly) project.

## Installation
Install via pip.

```shell
pip install git+https://github.com/amitness/drosophila
```

## Examples

### Apply for dimensionality reduction

```python
import numpy as np
from drosophila import FFA

x = np.random.randn(5, 768)
ffa = FFA(kc_size=50)
x_ffa = ffa.fit_transform(x)
x_ffa.shape
```
```
(5, 50)
```

### Plug into scikit-learn pipelines

Let's take a subset of the 20newgroups dataset reduced to only two categories.

```python
from sklearn.datasets import fetch_20newsgroups

categories = ['talk.religion.misc',
              'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',
                                     categories=categories)
x_train = newsgroups_train.data
y_train = newsgroups_train.target
x_test = newsgroups_test.data
y_test = newsgroups_test.target
```

The `FFA()` estimator can be used in all the places where dimensionality reduction techniques like `PCA()` are compatible.

```diff
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
- from sklearn.decomposition import PCA
+ from drosophila import FFA

vectorizer = make_pipeline(CountVectorizer(),
-                          PCA()
+                          FFA()
                          )
x_vec = vectorizer.fit_transform(x_train) # (961, 100)
```

Thus, you can use it to vectorize text documents into a compact representation.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from drosophila import FFA

vectorizer = make_pipeline(CountVectorizer(),
                           FFA())
x_vec = vectorizer.fit_transform(x_train) # (961, 100)
```

You can also plug it as part of text-classification pipelines.
```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from drosophila import FFA

pipeline = make_pipeline(CountVectorizer(),
                         FFA(kc_size=8853, wta=15, proj_size=10),
                         LogisticRegression())
pipeline.fit(x_train, y_train)
print('Train Accuracy:', pipeline.score(x_train, y_train))
y_pred = pipeline.predict(x_test)
print('Test F1:', f1_score(y_test, y_pred, average='macro'))
```
```
Train Accuracy: 1.0
Test F1: 0.9554628844483507
```

## Testing
To run the test for the package, run
```shell
pytest
````
