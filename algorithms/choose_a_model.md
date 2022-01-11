# Choosing the Right Machine Learning Algorithm

## Understand Your Data


The type and kind of data we have plays a key role in deciding which algorithm to use. Some algorithms can work with smaller sample sets while others require tons and tons of samples. Certain algorithms work with certain types of data. E.g. Naïve Bayes works well with categorical input but is not at all sensitive to missing data.

### Know your data

1. Look at Summary statistics and visualizations

Percentiles can help identify the range for most of the data
Averages and medians can describe central tendency
Correlations can indicate strong relationships

2. Visualize the data

Box plots can identify outliers
Density plots and histograms show the spread of data
Scatter plots can describe bivariate relationships

### Clean your data

- Deal with missing value. Missing data affects some models more than others. Even for models that handle missing data, they can be sensitive to it (missing data for certain variables can result in poor predictions)
- Choose what to do with outliers
    - Outliers can be very common in multidimensional data.
    - Some models are less sensitive to outliers than others. Usually tree models are less sensitive to the presence of outliers. However regression models, or any model that tries to use equations, could definitely be effected by outliers.
    - Outliers can be the result of bad data collection, or they can be legitimate extreme values.
- Does the data needs to be aggregated

### Augment your data

- Feature engineering is the process of going from raw data to data that is ready for modeling. It can serve multiple purposes:
    - Make the models easier to interpret (e.g. binning)
    - Capture more complex relationships (e.g. NNs)
    - Reduce data redundancy and dimensionality (e.g. PCA)
    - Rescale variables (e.g. standardizing or normalizing)
- Different models may have different feature engineering requirements. Some have built in feature engineering.

## Categorize the Problem

1. Categorize by input:
If you have labelled data, it’s a supervised learning problem.
If you have unlabelled data and want to find structure, it’s an unsupervised learning problem.
If you want to optimize an objective function by interacting with an environment, it’s a reinforcement learning problem.
2. Categorize by output.

If the output of your model is a number, it’s a regression problem.
If the output of your model is a class, it’s a classification problem.
If the output of your model is a set of input groups, it’s a clustering problem.
Do you want to detect an anomaly ? That’s anomaly detection

## Understand your constraints
What is your data storage capacity? Depending on the storage capacity of your system, you might not be able to store gigabytes of classification/regression models or gigabytes of data to clusterize. This is the case, for instance, for embedded systems.

Does the prediction have to be fast? In real time applications, it is obviously very important to have a prediction as fast as possible. For instance, in autonomous driving, it’s important that the classification of road signs be as fast as possible to avoid accidents.

Does the learning have to be fast? In some circumstances, training models quickly is necessary: sometimes, you need to rapidly update, on the fly, your model with a different dataset.

## Find the available algorithms
Now that you a clear understanding of where you stand, you can identify the algorithms that are applicable and practical to implement using the tools at your disposal. Some of the factors affecting the choice of a model are:

- Whether the model meets the business goals
- How much pre processing the model needs
- How accurate the model is
- How explainable the model is
- How fast the model is: How long does it take to build a model, and how long does the model take to make predictions.
- How scalable the model is

An important criteria affecting choice of algorithm is model complexity. Generally speaking, a model is more complex is:

- It relies on more features to learn and predict (e.g. using two features vs ten features to predict a target)
- It relies on more complex feature engineering (e.g. using polynomial terms, interactions, or principal components)
- It has more computational overhead (e.g. a single decision tree vs. a random forest of 100 trees).

Besides this, the same machine learning algorithm can be made more complex based on the number of parameters or the choice of some hyperparameters. For example,

- A regression model can have more features, or polynomial terms and interaction terms.
- A decision tree can have more or less depth.

Making the same algorithm more complex increases the chance of overfitting.

## Commonly Used Machine Learning Algorithms

![choosing the right model](https://scikit-learn.org/stable/_static/ml_map.png)

### Linear Regression

Linear Regression is used for predicting continuous values, for example, predicting the height of a person, based on their weight, hair size, age etc. It’s a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). More specifically, that y can be calculated from a linear combination of the input variables (x).

It is basically used to showcase the relationship between dependent and independent variables and show what happens to the dependent variables when changes are made to independent variables. requires minimal tuning
 
Linear Regressions are however unstable in case features are redundant, i.e. if there is multicollinearity.

Some examples where linear regression can used are:
- sales forecasting
- risk assessment analysis in health insurance companies
- Time to go one location to another
- Impact of blood alcohol content on coordination
- Predict monthly gift card sales and improve yearly revenue projections

### Logistic Regression

Logistic regression is emphatically not a classification algorithm on its own. Logistic regression is a regression model because it estimates the probability of class membership as a (transformation of a) multilinear function of the features.
Logistic regression is used for predicting variables which have only limited values. For example
1. To Identifying risk factors for diseases and planning preventive measures 
2. Classifying words as nouns, pronouns, and verbs.
3. Weather forecasting applications for predicting rainfall and weather conditions
4. In voting applications to find out whether voters will vote for a particular candidate or not

A good example of logistic regression is when credit card companies develop models which decide whether a customer will default on their loan EMIs or not.
The best part of logistic regression is that we can include more explanatory (dependent) variables such as dichotomous, ordinal and continuous variables to model binomial outcomes.
Logistic Regression is a statistical analysis (not classification) technique which is used for predictive analysis. It uses binary classification to reach specific outcomes and models the probabilities of default classes.
For a multilabel prediction, if we have a lot of labels, it is advisable to use Logistic Regression. For example, tag prediction problem for a question. A question can have any tag, in set of all tags. For such classification, we use logistic regression, as we need to build a lot of models, and a faster algorithm would help in such cases.



Logistic regression performs binary classification, so the label outputs are binary. It takes linear combination of features and applies non-linear function (sigmoid) to it, so it’s a very small instance of neural network.

Logistic regression provides lots of ways to regularize your model, and you don’t have to worry as much about your features being correlated, like you do in Naive Bayes. You also have a nice probabilistic interpretation, and you can easily update your model to take in new data, unlike decision trees or SVMs. Use it if you want a probabilistic framework or if you expect to receive more training data in the future that you want to be able to quickly incorporate into your model. Logistic regression can also help you understand the contributing factors behind the prediction, and is not just a black box method.

Logistic regression can be used in cases such as:

Predicting the Customer Churn
Credit Scoring & Fraud Detection
Measuring the effectiveness of marketing campaigns

### Decision trees

Applications of this Decision Tree Machine Learning Algorithm range from data exploration, pattern recognition, option pricing in finances and identifying disease and risk trends.
We want to buy a video game DVD for our best friend’s birthday but aren’t sure whether he will like it or not. We ask the Decision Tree Machine Learning Algorithm, and it will ask we a set of questions related to his preferences such as what console he uses, what is his budget. It’ll also ask whether he likes RPG or first-person shooters, does he like playing single player or multiplayer games, how much time he spends gaming daily and his track record for completing games.

Its model is operational in nature, and depending on our answers, the algorithm will use forward, and backward calculation steps to arrive at different conclusions.

Single trees are used very rarely, but in composition with many others they build very efficient algorithms such as Random Forest or Gradient Tree Boosting.

Decision trees easily handle feature interactions and they’re non-parametric, so you don’t have to worry about outliers or whether the data is linearly separable. One disadvantage is that they don’t support online learning, so you have to rebuild your tree when new examples come on. Another disadvantage is that they easily overfit, but that’s where ensemble methods like random forests (or boosted trees) come in. Decision Trees can also take a lot of memory (the more features you have, the deeper and larger your decision tree is likely to be)

Trees are excellent tools for helping you to choose between several courses of action.

Investment decisions
Customer churn
Banks loan defaulters
Build vs Buy decisions
Sales lead qualifications

### Random Forest

The random forest algorithm is used in industrial applications such as finding out whether a loan applicant is low-risk or high-risk, predicting the failure of mechanical parts in automobile engines and predicting social media share scores and performance scores.

The Random Forest ML Algorithm is a versatile supervised learning algorithm that’s used for both classification and regression analysis tasks. It creates a forest with a number of trees and makes them random. Although similar to the decision trees algorithm, the key difference is that it runs processes related to finding root nodes and splitting feature nodes randomly.

It essentially takes features and constructs randomly created decision trees to predict outcomes, votes each of them and consider the outcome with the highest votes as the final prediction.

### K-means

K-Means Clustering Algorithm is frequently used in applications such as grouping images into different categories, detecting different activity types in motion sensors and for monitoring whether tracked data points change between different groups over time. There are business use cases of this algorithm as well such as segmenting data by purchase history, classifying persons based on different interests, grouping inventories by manufacturing and sales metrics, etc.

when there is a large group of users and you want to divide them into particular groups based on some common attributes.

If there are questions like how is this organized or grouping something or concentrating on particular groups etc. in your problem statement then you should go with Clustering.

The biggest disadvantage is that K-Means needs to know in advance how many clusters there will be in your data, so this may require a lot of trials to “guess” the best K number of clusters to define.

### Principal component analysis (PCA)

Principal component analysis provides dimensionality reduction. Sometimes you have a wide range of features, probably highly correlated between each other, and models can easily overfit on a huge amount of data. Then, you can apply PCA.

One of the keys behind the success of PCA is that in addition to the low-dimensional sample representation, it provides a synchronized low-dimensional representation of the variables. The synchronized sample and variable representations provide a way to visually find variables that are characteristic of a group of samples.

---

PCA algorithm is used in applications such as gene expression analysis, stock
market predictions and in pattern classification tasks that ignore class labels.
The Principal Component Analysis (PCA) is a dimensionality reduction algorithm, used for speeding up learning algorithms and can be used for making compelling visualizations of complex datasets. It identifies patterns in data and aims to make correlations of variables in them. Whatever correlations the PCA finds is projected on a similar (but smaller) dimensional subspace.


### Support Vector Machines

SVM’s are very powerful supervised learning algorithm, and the way it works is by classifying data sets into different classes through a hyperplane. It marginalizes the classes and maximizes the distances between them to provide unique distinctions. We can use this algorithm for classification tasks that require more accuracy and efficiency of data.

Support Vector Machine (SVM) is a supervised machine learning technique that is widely used in pattern recognition and classification problems — when your data has exactly two classes.

High accuracy, nice theoretical guarantees regarding overfitting, and with an appropriate kernel they can work well even if you’re data isn’t linearly separable in the base feature space. Especially popular in text classification problems where very high-dimensional spaces are the norm. SVMs are however memory-intensive, hard to interpret, and difficult to tune.

SVM can be used in real-world applications such as:

detecting persons with common diseases such as diabetes
hand-written character recognition
text categorization — news articles by topics
stock market price prediction

Support Vector Machine Learning Algorithm is used in business applications such as comparing the relative performance of stocks over a period of time. These comparisons are later used to make wiser investment choices.

But it is advisable not to use SVM (with rbf kernel) for a large dataset.

### Naive Bayes

It is a classification technique based on Bayes’ theorem and very easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods. Naive Bayes is also a good choice when CPU and memory resources are a limiting factor.

Naive Bayes is super simple, you’re just doing a bunch of counts. If the NB conditional independence assumption actually holds, a Naive Bayes classifier will converge quicker than discriminative models like logistic regression, so you need less training data. And even if the NB assumption doesn’t hold, a NB classifier still often does a great job in practice. A good bet if want something fast and easy that performs pretty well. Its main disadvantage is that it can’t learn interactions between features.

Naive Bayes can be used in real-world applications such as:

Sentiment analysis and text classification
Recommendation systems like Netflix, Amazon
To mark an email as spam or not spam
Face recognition
document classification: classify web pages, forum posts, blog snippets, and tweets without manually going through them
disease prediction

ranking pages, indexing relevancy scores and classifying data categorically.


### Random Forest
Random Forest is an ensemble of decision trees. It can solve both regression and classification problems with large data sets. It also helps identify most significant variables from thousands of input variables. Random Forest is highly scalable to any number of dimensions and has generally quite acceptable performances. Then finally, there are genetic algorithms, which scale admirably well to any dimension and any data with minimal knowledge of the data itself, with the most minimal and simplest implementation being the microbial genetic algorithm. With Random Forest however, learning may be slow (depending on the parameterization) and it is not possible to iteratively improve the generated models

Random Forest can be used in real-world applications such as:

Predict patients for high risks
Predict parts failures in manufacturing
Predict loan defaulters

### Neural networks
Neural Networks take in the weights of connections between neurons . The weights are balanced, learning data point in the wake of learning data point . When all weights are trained, the neural network can be utilized to predict the class or a quantity, if there should arise an occurrence of regression of a new input data point. With Neural networks, extremely complex models can be trained and they can be utilized as a kind of black box, without playing out an unpredictable complex feature engineering before training the model. Joined with the “deep approach” even more unpredictable models can be picked up to realize new possibilities. E.g. object recognition has been as of late enormously enhanced utilizing Deep Neural Networks. Applied to unsupervised learning tasks, such as feature extraction, deep learning also extracts features from raw images or speech with much less human intervention.

On the other hand, neural networks are very hard to just clarify and parameterization is extremely mind boggling. They are also very resource and memory intensive.

Convolutional Neural Networks are feed-forward Neural networks which take in fixed inputs and give fixed outputs. For example — image feature classification and video processing tasks.
Recurrent Neural Networks use internal memory and are versatile since they take in arbitrary length sequences and use time-series information for giving outputs. For example — language processing tasks and text and speech analysis

### K-Nearest Neighbors Algorithm

KNN algorithm is used in industrial applications in tasks such as when a user wants to look for similar items in comparison to others. It’s even used in handwriting detection applications and image/video recognition tasks.

The best way to advance our understanding of these algorithms is to try our hand in image classification, stock analysis, and similar beginner data science projects.

The K-Nearest Neighbors Algorithm is a lazy algorithm that takes a non- parametric approach to predictive analysis. If we have unstructured data or lack knowledge regarding the distribution data, then the K-Nearest Neighbors Algorithm will come to our rescue. The training phase is pretty fast, and there is a lack of generalization in its training processes. The algorithm works by finding similar examples to our unknown example, and using the properties of those neighboring examples to estimate the properties of our unknown examples.

The only downside is its accuracy can be affected as it is not sensitive to outliers in data points.

### Recommender System

The Recommender Algorithm works by filtering and predicting user ratings and preferences for items by using collaborative and content-based techniques. The algorithm filters information and identifies groups with similar tastes to a target user and combines the ratings of that group for making recommendations to that user. It makes global product-based associations and gives personalized recommendations based on a user’s own rating.
For example, if a user likes the TV series ‘The Flash’ and likes the Netflix channel, then the algorithm would recommend shows of a similar genre to the user.