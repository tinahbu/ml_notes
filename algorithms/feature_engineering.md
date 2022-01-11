# Feature Engineering

## Feature Selection

- benefits 
    - smaller dimensionality
    - faster model training
    - reduces the model complexity, easier to interpret
    - improves the accuracy
    - reduces overfitting
- what features & why
- any situations this feature will not reflect my desired outcome?

### Filter Methods 

considers the relationship between features and the target variable to compute the importance of features

- Mutual Information
    - measures the dependence between random variables
    - if independent, mutual information is $0$
    - if $X$ is a deterministic function of $Y$, then mutual information is $1$
    - we can select our features by ranking their mutual information with the target variable
    - does well with the non-linear relationship between feature and target variable
- Pearson’s Correlation
    - quantifying linear dependence between two continuous variables X and Y
    - value varies from $-1$ to $+1$
    - $ρ_{X,Y} = \dfrac{cov(X, Y)}{\sigma_x \sigma_y}$ where
        - ${cov}$ is the covariance
        - $\sigma_x$ is the standard deviation of $X$
        - $\sigma_Y$ is the standard deviation of $Y$
- LDA
    - Linear discriminant analysis 
    - to find a linear combination of features that characterizes or separates two or more classes (or levels) of a categorical variable
- ANOVA
    - Analysis of variance
    - similar to LDA except for the fact that it is operated using one or more categorical independent features and one continuous dependent feature
    - provides a statistical test of whether the means of several groups are equal or not
- Chi-Square
    - a statistical test applied to the groups of categorical features to evaluate the likelihood of correlation or association between them using their frequency distribution
- Variance threshold
    - This method removes features with variation below a certain cutoff
    - The idea is when a feature doesn’t vary much within itself, it generally has very little predictive power.
    - doesn't consider the relationship of features with the target variable

### Wrapper Methods

generate models with a subsets of feature and gauge their model performances.

- forward selection
    - iterative
    - For data with n features, we create n models with each of the feature, select the best predictive feature. then create n-1 models with each of the remaining feature, select the best second feature. repeat until a subset of m features are selected/adding new feature doesnt improve the model performance.
- backward elimination
    - start with all the features and removes the least significant feature at each iteration
- greedy search, expensive to train so many models

### Embedded Methods

- inbuilt penalization functions
- Lasso regression
    - L1 regularization: absolute value of the coefficients
    - preventing overfitting also reduces the coefficients of less important features to zero
- Tree based models
    - tree models calculate feature importance for they need to keep the best performing features as close to the root of the tree. Constructing a decision tree involves calculating the best predictive feature.
    - The feature importance in tree based models are calculated based on Gini Index, Entropy or Chi-Square value

## Dimensionality Reduction

### Principal Component Analysis (PCA)

Transforms a large set of variables into a smaller one that still contains most of the information in the large set. data compression

- find k vectors onto which to project the data with min projection error (linear subspace)
- Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables.
- These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. 
- the principal components are less interpretable and don’t have any real meaning since they are constructed as linear combinations of the initial variables.
- principal components represent the directions of the data that explain a **maximal amount of variance**
- math
    - standardization $Z$
        - $x' = (x - mean)/std$ for each varialbe $x$
        - aim is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis
        - PCA is quite sensitive regarding the variances of the initial variables. variables with large ranges will dominate over those with small ranges, which will lead to biased results.
    - Compute the covariance matrix: $Z^T*Z$
        - The aim of this step is to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. 
        - variables are highly correlated in such a way that they contain redundant information. 
    - Calculate the eigenvectors and their corresponding eigenvalues of the covariance matrix
        - decompose $Z^T*Z$ into $PDP^{-1}$
        - where $P$ is the matrix of eigenvectors and $D$ is the diagonal matrix with eigenvalues on the diagonal and values of zero everywhere else.
        - engine vectors are independent of one another
        - SVD
            $[U, S, V] = svd(\sigma)$ where $U$ is the eigenvectors
            principal components are constructed in such a manner that the first principal component accounts for the largest possible variance in the data set.
            the line in which the projection of the points (red dots) is the most spread out. Or mathematically speaking, it’s the line that maximizes the variance (the average of the squared distances from the projected points (red dots) to the origin).
            the eigenvectors of the Covariance matrix are actually the directions of the axes where there is the most variance (most information) and that we call Principal Components. And eigenvalues are simply the coefficients attached to eigenvectors, which give the amount of variance carried in each Principal Component.
            By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance.
    - Feature vector
        - Take the eigenvalues $\lambda_1, \lambda_2, ..., \lambda_p$ and sort them from largest to smallest
        - take the first $k$ columns
        - discard those of lesser significance (of low eigenvalues), and form with the remaining ones a matrix of vectors that we call **feature vector**
    - Recast the data along the principal components axes
        - reorient the data from the original axes to the ones represented by the principal components
        - Calculate $Z^*$ = $ZP^*$. This new matrix, $Z^*$, is a centered/standardized version of $X$ but now each observation is a combination of the original variables, where the weights are determined by the eigenvector. As a bonus, because our eigenvectors in $P$ are independent of one another, each column of $Z^*$ is also independent of one another!
- steps 
    - PCA on training set: $x_i \in \mathbb{R}^{10,000}$ -> $z_i \in \mathbb{R}^{1,000}$
    - learn a model with the new set of features (faster)
        - $h_\theta(z)=\frac{1}{1 + e^{-\theta^Tz}}$
    - transform cross-validation/test set: $x$ -> $z$ -> $h_\theta(z)$

- PCA can't prevent overfitting
- What is the difference between logistic regression and PCA?
- What are the two pre-processing steps that should be applied before doing PCA?

- singular value decomposition
    - Why SVD instead of diagolization?
    - using the SVD to perform PCA makes much better sense numerically than forming the covariance matrix to begin with, since the formation of **𝐗****𝐗**⊤XX⊤ can cause loss of precision.

- sammon’s mapping



- density estimation

- recommender systems

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
- dummy encoding vs one-hot encoding
    - dummy encoding
        - n level categorical variable -> n-1 dummy variables
        - can include an intercept
    - one hot encoding
        - n level categorical variable -> n variables
        - perfect multi-collinearity -> big issue in linear regression
			- unsolvable for linear regression
			- need to set intercept to be false 

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

### Crossed Feature/Conjunction/Feature Interaction

- create a new feature by joining 2 categorical features 
    - e.g. lat * lon to create blocks for uber
    - e.g. user location * job title for linkedin job rec
- advantages
    - increased number of categories, A (20 categories) * B (10 categories) -> new feature A_B with 200 categories
    - the sample distribution is more finely delineated
    - introduced non-linearity, increases the predictive power of model 
- most commonly used: second-order interaction

    ```
    df["fea1_fea2"] = df["fea1"].astype('str')+"_"+df["fea2"].astype('str')
    ```

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

### Auto-encoders

- representation learning

## Embedding

- encode/transform entities in a lower dimensional vector space to capture their semantic information
- which helps to identify related entities (they'll be closer to each other in the vector space)
- dense multi-dimensional representation
- usually generated with neural networks
- enables transfer learning
- Context-based Embedding
	- one word may have different meanings in different contexts
- industry use cases
    - e.g. Twitter uses embeddings learned with general user interaction to serve more relevant ads
	- e.g. Doordash uses Store Embedding (store2vec) to personalize the store feed. Similar to word2vec, each store is one word and each sentence is one user session. Then, to generate vectors for a consumer, we sum the vectors for each store they ordered from in the past 6 months or a total of 100 orders. Then, the distance between a store and a consumer is determined by taking the cosine distance between the store’s vector and the consumer’s vector.
	- e.g. Similarly, Instagram uses account embedding to recommend relevant content (photos, videos, and Stories) to users.
- recommend: $d = \sqrt[4]{D}$ Where D is the number of categories. Another way is to treat $d$ as a hyperparameter

### `word2vector`

- word embedding
- shallow neutral network with 1 hidden layer
- self-supervised
- **CBOW**: Continuous bag of words (CBOW) tries to predict the current word from its surrounding words by optimizing for following loss function: $Loss = -log(p(word_t|word_{t-n}, ..., word_{t-1}, word_{t+1}, ... word_{t+n}))$ where $n$ is the size of the window
![cbow](https://user-images.githubusercontent.com/7570863/129840508-d43bf41a-2dc7-4430-b84f-12df3842caad.png)
- **Skipgram**: predict surrounding words from the current word: $Loss = -log(p(word_{t-n}, ..., word_{t-1}, word_{t+1}, ..., word_{t+n}|word_t))$
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

- $x' = (x - min)/(max - min)$
    - mean $ = 0$
    - values between $[0, 1]$
- shifted & rescaled
- important for distance based classifiers like SVM, KNN, MLP
- trees are agnostic to scaling

### Standardization

- $x' = (x - mean)/std$
    - centered around the mean with a unit standard deviation
    - mean $ = 0$
    - std $ = 1$ (unit standard deviation)
- use when some feature with larger values can dominate completely the others with smaller values
- will not affect outliers
