# Feature Engineering

## Feature Selection
- what features & why
- any situations this feature will not reflect my desired outcome?

## Encoding

Because ML algorithms can only handle numerical features. 
Need to be performed both on the train and test sets.

example data:

```
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd

data = np.array(["Sun", "Sun", "Moon", "Earth", "Monn", "Venus"])
df = pd.DataFrame({"col1": ["Sun", "Sun", "Moon", "Earth", "Moon", "Venus"]})
```

### Label Encoding
- each category to its individual integer
- use when categorical feature is ordinal (rank among the categories)
- `sklearn` `LabelEncoder` object/function
    
    ```
    labelenc = preprocessing.LabelEncoder()
    labelenc.fit(data)
    targets_trans = labelenc.transform(data)
    # transformed to [3 3 2 0 1 4]
    ```
    
- `pandas` `category` data type
    
    ```
    # df.dtypes originally are object
    df["col1"] = df["col1"].astype("category")
    df["col1_label_encoding"] = df["col1"].cat.codes
    #     col1  col1_label_encoding
    # 0    Sun                    3
    # 1    Sun                    3
    # 2   Moon                    2
    # 3  Earth                    0
    # 4   Monn                    1
    # 5  Venus                    4
    ```
    
### One-Hot Encoding
- categorical data -> one-hot numeric array with 1s and 0s
- use when categorical feature is not ordinal (no rank)
- each category will be added as a feature
- one-hot encoding is the process of creating dummy variables
- need to drop a column to avoid multicollinearity
- medium cardinality: curse of dimensionality
- some categories are not important, can be grouped together into the "other" class
- `sklearn` `OneHotEncoder` object/function 
 
    ```
    labelEnc = preprocessing.LabelEncoder()
    new_data = labelEnc.fit_transform(data)
    onehotEnc = preprocessing.OneHotEncoder()
    onehotEnc.fit(new_data(-1, 1))
    data_trans = onehotEnc.transform(new_data.reshape(-1, 1))
    # [[0. 0. 1. 0.]
    # [0. 0. 1. 0.]
    # [0. 1. 0. 0.]
    # [1. 0. 0. 0.]
    # [0. 1. 0. 0.]
    # [0. 0. 0. 1.]]
    ```

- `pandas` `get_dummies` function

    ```
    df_new = pd.get_dummies(df, columns=["col1"], prefix="Planet")
    #    Planet_Earth  Planet_Moon  Planet_Sun  Planet_Venus
    # 0             0            0           1             0
    # 1             0            0           1             0
    # 2             0            1           0             0
    # 3             1            0           0             0
    # 4             0            1           0             0
    # 5             0            0           0             1
    ```
    
### Feature Hashing
- categorical data/text data -> feature vector of arbitrary dimensionality
- high cardinality
- allow multiple values to be encoded as the same value
- problem: collision
### Crossed Feature/Conjunction
- e.g. lat * lon to create blocks for uber
- e.g. user location * job title for linkedin job rec
### Count Encoding
- use the number of occurences of a category as its encoding
- 
    ```
    df["planet_count"] = df["col1"].map(df["col1"].value_counts().to_dict())
    ```
    
- disadvantage
    - not friendly to new features
    - easy to get conflicts between features
- popular on Kaggle for tree-based models like xgboost 
### Mean Encoding
- use the mean of any numerical features as encoding

    ```
    d = df.groupby(["col1"])["price"].mean().to_dict()
    df["col1_price_mean"] = df["col1"].map(d)
    ```
    
- similarly, can also do std encoding, var encoding, max encoding, etc
- usually used for classification tasks
### Weight of Evidence (WOE) Encoding
- measure of evidence on one side of an issue compared with evidence on the other side of the issue
- binary classification: WOE = ln(p(1)/p(0))
    /weigth_of_evidence_encoding.png

    ```
    df = pd.DataFrame({
        "col1": ["Moon", "Sun", "Moon", "Sun", "Sun"],
        "Target": [1, 1, 0, 1, 0]
    })
    df["Target"] = df["Target"].astype("float64")
    d = df.groupby(["col1"])["Target"].mean().to_dict()
    df["p1"] = df["col1"].map(d)
    df["p0"] = 1 - df["p1"]
    df["woe"] = np.log(df["p1"] / df["p0"])
    ```
    
## Embedding
- encode/transform entities in a lower dimensional vector space to capture their semantic information
- which helps to identify related entities (they'll be closer to each other in the vector space)
- dense multi-dimensional representation
- e.g. Twitter uses Embedding for UserIDs and cases like recommendations, nearest neighbor searches, and transfer learning.
	- e.g. Doordash uses Store Embedding (store2vec) to personalize the store feed. Similar to word2vec, each store is one word and each sentence is one user session. Then, to generate vectors for a consumer, we sum the vectors for each store they ordered from in the past 6 months or a total of 100 orders. Then, the distance between a store and a consumer is determined by taking the cosine distance between the store’s vector and the consumer’s vector.
	- e.g. Similarly, Instagram uses account embedding to recommend relevant content (photos, videos, and Stories) to users.
- usually generated with neural networks
- recommend: d = \sqrt[4]{D} Where D is the number of categories. Another way is to treat d as a hyperparameter
- enable transfer learning
	- Twitter uses embeddings learned with general user interaction to serve more relevant ads.
- Context-based Embedding
	- one word may have different meanings in different contexts

### `word2vector`

- word embedding
- shallow neutral networks with 1 hidden layer
- self-supervised
- **CBOW**: Continuous bag of words (CBOW) tries to predict the current word from its surrounding words by optimizing for following loss function: Loss = -log(p(word_t|word_t-n, ..., word_t-1, word_t+1, ... word_t+n)) where n is the window
![cbow](https://user-images.githubusercontent.com/7570863/129840508-d43bf41a-2dc7-4430-b84f-12df3842caad.png)
- **Skipgram**: predict surrounding words from the current word: `Loss = -log(p(word_t-n, ..., word_t-1, word_t+1, ... word_t+n|word_t))`
![skipgram](https://user-images.githubusercontent.com/7570863/129840616-b28b2b60-8446-4f7c-baa1-b2e60f7d0498.png)
- to predict whether a user is interested in a particular document given the documents that they have previously read: represent the user by taking the mean of the Word2vec embeddings of document titles that they have engaged with. Similarly, we can represent the document by the mean of its title term embeddings. We can simply take the dot product of these two vectors and use that in our ML model.


### Image embedding
- Auto-encoders use neural networks consisting of both an encoder and a decoder. They first learn to compress the raw image pixel data to a small dimension via an encoder model and then try to de-compress it via a decoder to re-generate the same input image. The last layer of encoder determines the dimension of the embedding, which should be sufficiently large to capture enough information about the image so that the decoder can decode it.
- Once we have trained the model, we only use the encoder (first N network layers) to generate embeddings for images.
- image search problem: when we want to find the best images for given text terms, e.g. query “cat images”. In this case, image embedding along with query term embedding can help refine search relevance models.

### User & Item embedding
- generate embeddings for them in the same space with a two-tower neural network
- The model optimizes the inner product loss such that positive pairs from entity interactions have a higher score and random pairs have a lower score
![image](https://user-images.githubusercontent.com/7570863/129841511-8a7431b7-66dc-4487-8050-ca6c841e9768.png)

## Feature Interaction
- create a new feature by joining 2 categorical features 
- advantages
    - increased number of categories, A (20 categories) * B (10 categories) -> new feature A_B with 200 categories
    - the sample distribution is more finely delineated
    - introduced non-linearity, increases the predictive power of model 
- most commonly used: second-order interaction

    ```
    df["fea1_fea2"] = df["fea1"].astype('str')+"_"+df["fea2"].astype('str')
    ```
    
## Datetime Features

```
df = pd.DataFrame({"col1": [1549720105, 1556744905, 1569763805, 1579780105]})  # raw UNIX timestamp
df["col1"] = pd.to_datetime(df['col1'], unit='s')
df["month"] = df["col1"].dt.month
df["week"] = df["col1"].dt.week
df["hour"] = df["col1"].dt.hour

"""
                 col1  month  week  hour
0 2019-02-09 13:48:25      2     6    13
1 2019-05-01 21:08:25      5    18    21
2 2019-09-29 13:30:05      9    39    13
3 2020-01-23 11:48:25      1     4    11
"""
```

## Numeric Features

### Normalization
- mean = 0
- values [0, 1]
- x' = (x - min)/(max - min)
- shifted & rescaled
- important for distance based classifiers like SVM, KNN, MLP
- trees are agnostic to scaling

### Standardization
- centered around the mean with a unit standard deviation
- mean = 0
- std = 1 (unit standard deviation)
- x' = (x - mean)/std
- use when some feature with larger values can dominate completely the others with smaller values
- will not affect outliers

## Feature store

- feature pipelines
- feed to inference