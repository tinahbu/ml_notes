# TODO - some basics questions
Can you state Tom Mitchell’s definition of learning and discuss T, P and E?
What can be different types of tasks encountered in Machine Learning?
What are supervised, unsupervised, semi-supervised, self-supervised, multi-instance learning, and reinforcement learning?
Loosely how can supervised learning be converted into unsupervised learning and vice-versa?
Consider linear regression. What are T, P and E?
Derive the normal equation for linear regression.
What do you mean by affine transformation? Discuss affine vs. linear transformation.
Discuss training error, test error, generalization error, overfitting, and underfitting.
Compare representational capacity vs. effective capacity of a model.
Discuss VC dimension.
What are nonparametric models? What is nonparametric learning?
What is an ideal model? What is Bayes error? What is/are the source(s) of Bayes error occur?
What is the no free lunch theorem in connection to Machine Learning?

What is weight decay? What is it added?
What is a hyperparameter? How do you choose which settings are going to be hyperparameters and which are going to be learned?

What are point estimation and function estimation in the context of Machine Learning? What is the relation between them?
What is the maximal likelihood of a parameter vector $theta$? Where does the log come from?
Prove that for linear regression MSE can be derived from maximal likelihood by proper assumptions.
Why is maximal likelihood the preferred estimator in ML?
Under what conditions do the maximal likelihood estimator guarantee consistency?

Given a black box machine learning algorithm that you can’t modify, how could you improve its error? (you can transform the input for example.)
How to find the best hyper parameters? (Random search, grid search, Bayesian search (and what it is?))
What is transfer learning?



# Learning Theory


# Model and Feature Selection
Why are model selection methods needed?
How do you do a trade-off between bias and variance?
What are the different attributes that can be selected by model selection methods?
Why is cross-validation required?
Describe different cross-validation techniques.
What is hold-out cross validation? What are its advantages and disadvantages?
What is k-fold cross validation? What are its advantages and disadvantages?
What is leave-one-out cross validation? What are its advantages and disadvantages?
Why is feature selection required?
Describe some feature selection methods.
What is forward feature selection method? What are its advantages and disadvantages?
What is backward feature selection method? What are its advantages and disadvantages?
What is filter feature selection method and describe two of them?
What is mutual information and KL divergence?
Describe KL divergence intuitively.
# Curse of Dimensionality
Describe the curse of dimensionality with examples.
What is local constancy or smoothness prior or regularization?


What is Empirical Risk Minimization?
What is Union bound and Hoeffding’s inequality?
Write the formulae for training error and generalization error. Point out the differences.
State the uniform convergence theorem and derive it.
What is sample complexity bound of uniform convergence theorem?
What is error bound of uniform convergence theorem?
What is the VC dimension?
What does the training set size depend on for a finite and infinite hypothesis set? Compare and contrast.
What is the VC dimension for an n-dimensional linear classifier?
How is the VC dimension of a SVM bounded although it is projected to an infinite dimension?
Considering that Empirical Risk Minimization is a NP-hard problem, how does logistic regression and SVM loss work?

## Confidence Interval
What is population mean and sample mean?
What is population standard deviation and sample standard deviation?
Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.?
What is the formula for calculating the s.d. of the sample mean?
What is confidence interval?
What is standard error?

# Supervised Learning
## Regression
### Linear regression
- hypothesis:  hΘ(x) = WX + B
	- Categorical Variable
		- dummy encoding
			- n level categorical variable -\> n-1 dummy variables
			- A nth dummy variable is redundant; it carries no new information. And it creates a severe multicollinearity problem.
			- can include an intercept
		- one hot encoding
			- n level categorical variable -\> n variables
			- perfect multi-collinearity -\> big issue in linear regression
			- unsolvable for linear regression
			- need to set intercept to be false 
## Classification
### Logistic regression
- hypothesis: Z = WX + B, hΘ(x) = sigmoid (Z)
	- linear classifier (or single layer perceptron)
- sigmoid function: 1 / (1 + e^-z)
	- (-∞,+∞) -\> (0, 1)
	- predictions -\> probabilities
	- ![img][image-3]
- Loss function
	- can't use MSE because with sigmoid, MSE results in a non-convex function with many local minimums
	- cross entropy loss
		- ![loss function][image-4]
		- The key thing to note is the cost function penalizes confident and wrong predictions more than it rewards confident and right predictions! The corollary is increasing prediction accuracy (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of our cost function.
		- ![\_images/y1andy2\_logistic\_function.png](/Users/Tina/Google Drive/ML-Interview/assets/y1andy2\_logistic\_function.png)
	- Cost Function
		- ![\_images/logistic\_cost\_function\_joined.png](/Users/Tina/Google Drive/ML-Interview/assets/logistic\_cost\_function\_joined.png)
- MLE
	- ![logistic\_regression\_mle](/Users/Tina/Google Drive/Job\_Application/c3.ai/logistic\_regression\_mle.png)
- Cons
	- can't apply to linearly separable data
		- max data likelihood, cost surface will have a surface
- multiclass logistic regression
	- Instead of \(y = {0,1}\) we will expand our definition so that \(y = {0,1...n}\). Basically we re-run binary classification multiple times, once for each class.
	- Steps
		- Divide the problem into n+1 binary classification problems (+1 because the index starts at 0?).
		- For each class…
		- Predict the probability the observations are in that single class.
		- prediction = max(probability of the classes)
		- For each sub-problem, we select one class (YES) and lump all the others into a second class (NO). Then we take the class with the highest predicted value.
### decision tree
- criteria
	- gini
	- information gain
	- entropy
- regularization -\> pruning
	- remove branches that have weak predictive power
	- to reduce model complexity and predictive accuracy
	- bottom-up or top-down
	- reduced error pruning: Starting at the leaves, each node is replaced with its most popular class. If the prediction accuracy is not affected then the change is kept. While somewhat naive, reduced error pruning has the advantage of simplicity and speed.
	- cost complexity pruning: [https://en.wikipedia.org/wiki/Decision\_tree\_pruning][1]
### SVM - Support Vector Machine
- Produces *nonlinear* boundaries by constructing a linear boundary in a large, transformed version of the feature space
- Maximum margin classifier
	- fit the maximum-margin hyperplane in a transformed feature space
	- large margin for all classes
	- a line L1 is said to be a better classifier than line L2, if the “margin” of L1 is larger i.e., L1 is farther from both classes.
	- Generalize better when margin is large
		- maximize the probability of classifying correctly unseen instances
		- minimize the expected generalization loss (instead of the expected empirical loss)
		-  only the nearest instances to the separator matter
	- ![SVM](/Users/Tina/Google Drive/ML-Interview/assets/3547\_03\_07.jpg)
- Soft margin classifier
	- allows some points in the training data to violate the separating line
	- slack variables - `ξi`
		- the number of misclassifications / violations of the margin
			- `ξi =0` is no violation and we are back to the inflexible Maximal-Margin Classifier 
			- the larger the valuee, the more violations of the hyperplane are permitted
		- Trade off between margin’s size and \#misclassifications in training set
	- Regularization `C`
		- ![][image-5]
		- larger -\> smaller margin, less regularization, less misclassification, more overfitting, bias
		- smaller -\> larger margin, more regularization, more misclassification, less overfitting, variance, smooth decision surface
		- C = 1/lambda, where lambda is the regularization parameter
		- Choose with cross validation
- support vector
	- The examples closest to the separator are support vectors
		- in the margin
	- The margin (ρ) of the separator is the distance between support vectors
		- $margin=2/\|w\|$
		- -\> maximizing the margin is the same that minimizing the norm of the weights
	- ![SVM support vector](/Users/Tina/Google Drive/ML-Interview/assets/slack.png)
- kernel trick
	- $K(x\_i, x\_j) = \varphi(x\_i)^T \varphi(x\_j)$
	- example
		- ![][image-6]
	- benefit
		- Working directly in the feature space can be costly
		- We have to explicitly create the feature space and operate in it
		- We may need infinite features to make a problem linearly separable
	- only the inner product of the examples is needed for SVM
	- need to define how to compute this product
		- many geometric operations that can be expressed as inner products, like angles or distances
		- Polynomial kernel of degree d: K(x, y) = (xT y + 1)d
		- Gaussian function with width σ: K(x, y) = exp(−||x − y||2/2σ2)
		- Sigmoid hiperbolical tangent with parameters k and θ: K(x, y) = tanh(kxT y + θ) (only for certain values of the parameters)
		-  linear
		- polynomial
			- (gamma⟨x,x′⟩+r)^d, d is specified by keyword degree, r by coef0.
		- rbf
			- A radial basis function (RBF) is a real-valued function whose value depends only on the distance from the origin; or the distance from some other point, called a center
			- exp⁡(−gamma‖x−x′‖^2)
			- usually Euclidean distance
			- The Gaussian kernel is a specific example of a radial basis function
		- sigmoid
			- tanh(gamma(x, x’) + r
	- Kernel functions can also be interpreted as similarity function
	- A similarity function can be transformed in a kernel given that hold the sdp matrix condition

	- feature transformation
	- transform to a higher dimensional space with a linear boundary
	- generalized inner product
	- kernel functions that enable in higher-dimension spaces without explicitly calculating the coordinates of points within that dimension
	- computational cheaper 
	- many algorithms can be expressed in terms of inner products
	- ![img](/Users/Tina/Google Drive/ML-Interview/assets/Nb155MpRK4po7rABnZUr5XX3jX2ZZAQ9osetXlWhTfwznZjZwJtwwhCnxzs-8xK\_NNg7xg8NdQbKdj2OZB5VrG-6jzvIDNW4v1UzlUqVEaKmk-LN80H9v06O-4QaZpxOnjLqhIsa.png)
- gamma
	- how close a point needs to be from the hyperplane to be included in the calculation
	- How much influence a single training example has
	- Support vector: subset of the training data
	- Controls the position and orientation of the hyperplane
	- How does the SVM kernel parameter sigma² affect the bias/variance trade off?
- loss function: hinge loss
	- Convex
	- Local optimal == global optimal
	- Not differentiable (not smooth) -\> can’t be used with gradient descent
	- Correctly classified points lying outside the margin boundaries of the support vectors are not penalized, whereas points within the margin boundaries or on the wrong side of the hyperplane are penalized in a linear fashion compared to their distance from the correct boundary.
	- ![hinge\_loss](/Users/Tina/Google Drive/ML-Interview/assets/hoaGW.png)
- outliers
	- SVM doesn’t handle outliers
	- using soft-margin helps
- How can the SVM optimization function be derived from the logistic regression optimization function?
\- 
In SVM, what is the angle between the decision boundary and theta?
What is the mathematical intuition of a large margin classifier?
How are the landmarks initially chosen in an SVM? How many and where?

Can we apply the kernel trick to logistic regression? Why is it not used in practice then?
What is the difference between logistic regression and SVM without a kernel?
Can any similarity function be used for SVM?
Logistic regression vs. SVMs: When to use which one?
### Naive Bayes
- naive: independence between features
	- Bayes Theorem
		- P(c|x) = P(x|c) P(c) / P(x)
		- P(A|B) = P(B|A)P(A)/P(B)
		- posterior = likelihood \* class prior / predictor prior
### KNN
- classifies new cases by a majority vote of its k neighbors
- lazy learner
	- does not have any learning involved, i.e., there are no parameters we can tune to make the performance better. Or we are not trying to optimize an objective function from the training data set.
	- all computation is deferred until classification
	- non-parametric method used for classification and regression
- Steps
	- initialize K to your chosen number of neighbors
	- for each example in the data
		- compute distance between the query example and the current example from the data
		- add the distance and the index of the example to an ordered collection
	- sort the ordered collection of distances from smallest to largest (ascending)
	- pick the first k entries, get the labels/values of the k entries
	- or assign weight to the contributions of all the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/*d*, where *d* is the distance to the neighbor.
	- if regression, return the mean; if classification return the most popular vote
- distance function
	- continuous
		- Euclidean
		- Manhattan
		- Minkowski
	- categorical
		- Hamming
	- requires normalization
- Cons
	- expensive O(N^2)
### Random Forest
- bagging method where deep tree, fitted on bootstrap samples, are combined to produce an output with lower variance. 
- classification & regression
- when growing each tree, instead of only sampling over the observations in the dataset to generate a bootstrap sample, we also sample over features and keep only a random subset of them to build the tree.
- search for the best features among a random subset of features (when splitting a node)
	- prevents overfitting, more robust
- train each tree independently, with a random sample of the data & features
- what is random
	- Random sampling of training data points when building trees
		- bootstrapping: drawn randomly with replacement
		- each tree might have high variance with respect to a particular set of the training data, but, the entire forest will have lower variance but not at the cost of increasing the bias.
	- Random subsets of features considered when splitting nodes
- each tree is grown to largest extent possible, no pruning
- feature importance
	- final feature importance = avg(feature inportance in all trees)
	- node impurities in each tree
- overfitting
	- number of samples in bootstrap
	- number of features in each tree
### Gradient Boosting
- Gradient boosting build trees one at a time, where each new tree helps to correct errors by previously trained tree (boosting of decision tree)
- GBM
- XGBoost
- LightGBM
	- faster training and lower memory usage
	- parallel and GPU supported
- CatBoost

# Unsupervised Learning

## Clusering
Describe the k-means algorithm.
What is distortion function? Is it convex or non-convex?
Tell me about the convergence of the distortion function.
Topic: EM algorithm
What is the Gaussian Mixture Model?
Describe the EM algorithm intuitively.
What are the two steps of the EM algorithm
Compare Gaussian Mixture Model and Gaussian Discriminant Analysis.

K-means++

## K-means
- *k*-means clustering aims to partition *n* observations into *k* clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 
- Steps
	- select k
		-  the number of centroids you need in the dataset
		- Choose k with the elbow method
			- run k-means clustering on the dataset for a range of values of *k* (say, *k* from 1 to 10 in the examples above), and for each value of *k* calculate the sum of squared errors (SSE). 
			- Then, plot a line chart of the SSE for each value of *k*. If the line chart looks like an arm, then the "elbow"/bend on the arm is the value of *k* that is the best.
			- The idea is that we want a small SSE, but that the SSE tends to decrease toward 0 as we increase *k* (the SSE is 0 when *k* is equal to the number of data points in the dataset, because then each data point is its own cluster, and there is no error between it and the center of its cluster). So our goal is to choose a small value of *k* that still has a low SSE, and the elbow usually represents where we start to have diminishing returns by increasing *k*.
			- ![Image result for elbow method][image-7]
			- However, the elbow method doesn't always work well; especially if the data is not very clustered. does not have a clear elbow. Instead, we see a fairly smooth curve, and it's unclear what is the best value of *k* to choose. 
	- Initialize center points
		- Forgy method: randomly choose k observations as centroids
		- Random Partition method: randomly assign a cluster to each observation and then proceed to the update step, thus computing the initial mean to be the centroid of the cluster's randomly assigned points
		- The Forgy method tends to spread the initial means out, while Random Partition places all of them close to the center of the data set. 
	- Assignment 
		- Assign each observation to the cluster whose mean has the least squared Euclidean distance, this is intuitively the "nearest" mean.
		- J (cost) decrease, holding centroids constant
	- Update 
		- Calculate the new means (centroids) of the observations in the new clusters.
		- J (cost) decrease, holding cluster assignment constant
	- Iterate
		- Repeat these steps for a set number of iterations or when the assignments no longer change. 
		- The algorithm does not guarantee to find the optimum
		- The result may depend on the initial clusters. -\> randomly initialize the group centers a few times, and then select the run that looks like it provided the best results.
- Optimize
	- total intra-cluster variation / total within cluster sum of square
	- distortion cost function
	- Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.
	- In other words, the K-means algorithm identifies *k* number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
- Pros
	- Fast training O(N)
- Cons
	- difficult to choose k
	- lack consistency: K-means starts with a random choice of cluster centers and therefore it may yield different clustering results on different runs of the algorithm
	- However, its performance is usually not as competitive as those of the other sophisticated clustering techniques because slight variations in the data could lead to high variance.
	- Mean-Shift Clustering
		- Mean shift clustering is a sliding-window-based algorithm that attempts to find dense areas of data points. It is a centroid-based algorithm meaning that the goal is to locate the center points of each group/class, which works by updating candidates for center points to be the mean of the points within the sliding-window. These candidate windows are then filtered in a post-processing stage to eliminate near-duplicates, forming the final set of center points and their corresponding groups. 
		- Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region on each step until convergence.
		- Steps
			- 1. We begin with a circular sliding window centered at a point C (randomly selected) and having radius r as the kernel.
			- 2. At every iteration the sliding window is shifted towards regions of higher density by shifting the center point to the mean of the points within the window (hence the name).
			- 3. We continue shifting the sliding window according to the mean until there is no direction at which a shift can accommodate more points inside the kernel.
			- This process of steps 1 to 3 is done with many sliding windows until all points lie within a window. When multiple sliding windows overlap the window containing the most points is preserved. The data points are then clustered according to the sliding window in which they reside.
		- ![img](/Users/Tina/Google Drive/ML-Interview/assets/1\*vyz94J\_76dsVToaa4VG1Zg.gif)
		- Pros
			- no need to select k
		- Cons
			- the selection of the window size/radius “r” can be non-trivial.
	- There are 3 more to read about [here][2]
- Frequent Itemset Mining
	- Apiori
### Dimensionality Reduction
- Why do we need dimensionality reduction techniques?
- What do we need PCA and what does it do?
    - data compression
    - visualization: k = 2 / 3
- What can't PCA do?
    - prevent overfitting: use regularization
    - run it before training the algorithm, using the original dataset
- Steps
    - PCA on training set: $x_i \in \mathbb{R}^10000$ -> $z_i \in \mathbb{R}^1000$
    - learn the model (faster)
    - $h_\theta(z)=\frac{1}{1 + e^{-\theta^Tz}}$
    - cross-validation/test set: $x$ -> $z$ -> $h_\theta(z)$

What is the difference between logistic regression and PCA?
What are the two pre-processing steps that should be applied before doing PCA?

- feature elimination
	- drop some variables
	- feature extraction
		- Principal Component Analysis (PCA)
			- transforming a large set of variables into a smaller one that still contains most of the information in the large set
			- find k vectors onto which to project the data with min projection error (linear subspace)
			- Principal components are new variables that are constructed as linear combinations or mixtures of the initial variables.
			- These combinations are done in such a way that the new variables (i.e., principal components) are uncorrelated and most of the information within the initial variables is squeezed or compressed into the first components. 
			- the principal components are less interpretable and don’t have any real meaning since they are constructed as linear combinations of the initial variables.
			- principal components represent the directions of the data that explain a **maximal amount of variance**
			- Steps
				- standardization `Z`
					- The aim of this step is to standardize the range of the continuous initial variables so that each one of them contributes equally to the analysis.
					- More specifically, the reason why it is critical to perform standardization prior to PCA, is that the latter is quite sensitive regarding the variances of the initial variables. That is, if there are large differences between the ranges of initial variables, those variables with larger ranges will dominate over those with small ranges, which will lead to biased results.
					- Mathematically, this can be done by subtracting the mean and dividing by the standard deviation for each value of each variable.
						![img][image-8]
				- Compute the covariance matrix: `Z^T*Z`
					- The aim of this step is to understand how the variables of the input data set are varying from the mean with respect to each other, or in other words, to see if there is any relationship between them. 
					- variables are highly correlated in such a way that they contain redundant information. 
						![img][image-9]
				- Calculate the eigenvectors and their corresponding eigenvalues of the covariance matrix
					- decompose **Z**ᵀ**Z** into **PDP**⁻¹
					- where **P** is the matrix of eigenvectors and **D** is the diagonal matrix with eigenvalues on the diagonal and values of zero everywhere else.
					- engine vectors are independent of one another
					- SVD
						[U, S, V] = svd(sigma) where U is the eigenvectors
						principal components are constructed in such a manner that the first principal component accounts for the **largest possible variance** in the data set.
						the line in which the projection of the points (red dots) is the most spread out. Or mathematically speaking, it’s the line that maximizes the variance (the average of the squared distances from the projected points (red dots) to the origin).
						the eigenvectors of the Covariance matrix are actually *the* *directions of the axes where there is the most variance* (most information) and that we call Principal Components. And eigenvalues are simply the coefficients attached to eigenvectors, which give the *amount of variance carried in each Principal Component*.
						By ranking your eigenvectors in order of their eigenvalues, highest to lowest, you get the principal components in order of significance.
				- Feature vector
					- Take the eigenvalues λ₁, λ₂, …, λ*p* and sort them from largest to smallest
					- take the first k columns
					- discard those of lesser significance (of low eigenvalues), and form with the remaining ones a matrix of vectors that we call *Feature vector*.
				- Recast the data along the principal components axes
					- reorient the data from the original axes to the ones represented by the principal components
					- Calculate **Z\*** = **ZP\***. This new matrix, **Z\***, is a centered/standardized version of **X** but now each observation is a combination of the original variables, where the weights are determined by the eigenvector. **As a bonus, because our eigenvectors in P\* are independent of one another, each column of Z\* is also independent of one another!** 
			- Why SVD instead of diagolization
				- using the SVD to perform PCA makes much better sense numerically than forming the covariance matrix to begin with, since the formation of **𝐗****𝐗**⊤XX⊤ can cause loss of precision.
- representation learning

	- auto-encoders

- density estimation

- recommender systems
### Bayesian Machine Learning
What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
Compare and contrast maximum likelihood and maximum a posteriori estimation.
How does Bayesian methods do automatic feature selection?
What do you mean by Bayesian regularization?
When will you use Bayesian methods instead of Frequentist methods?

### Reinforcement Learning

- Markov Decision Process

### NLP
What is WORD2VEC?
What is t-SNE? Why do we use PCA instead of t-SNE?
What is sampled softmax?
Why is it difficult to train a RNN with SGD?
How do you tackle the problem of exploding gradients?
What is the problem of vanishing gradients?
How do you tackle the problem of vanishing gradients?
Explain the memory cell of a LSTM.
What type of regularization do one use in LSTM?
What is Beam Search?
How to automatically caption an image?


- TFIDF
- tokenization

### Time Series
- cross-validation
	- validation set needs to come chronologically after the training subset
	- forward chaining
		- multiple train test sets
		- test sets only have 1 observation
		- corresponding training set consists of all the observations that occurred prior to the test observation (no future observation)
		- ![Image result for time series cross validation][image-10]
		- The forecast accuracy is computed by averaging over the test sets

### Ensemble

- Ensemble learning is a machine learning paradigm where multiple models (often called “weak learners”) are trained to solve the same problem and combined to get better results. The main hypothesis is that when weak models are correctly combined we can obtain more accurate and/or robust models.

	In ensemble learning theory, we call **weak learners** (or **base models**) models that can be used as building blocks for designing more complex models by combining several of them. Most of the time, these basics models perform not so well by themselves either because they have a high bias (low degree of freedom models, for example) or because they have too much variance to be robust (high degree of freedom models, for example). Then, the idea of ensemble methods is to try reducing bias and/or variance of such weak learners by combining several of them together in order to create a **strong learner** (or **ensemble model**) that achieves better performances.

- bagging

	- considers homogeneous weak learners, learns them independently from each other in parallel and combines them following some kind of deterministic averaging process
	- a single base learning algorithm is used
	- use bootstrap samples (representativity and independence) to fit models that are almost independent.
		- bootstrap: This statistical technique consists in generating samples of size B (called bootstrap samples) from an initial dataset of size N by randomly drawing with replacement B observations.
		- approximatively independent and identically distributed (i.i.d.)
	- average for regression / majority vote for classification
		- A key insight for ensembling predictors is that by averaging (or generally aggregating) many low bias, high variance predictors, we can reduce the variance while retaining the low bias. 
		- Each estimate is centered around the true density, but is overly complicated (low bias, high variance). By averaging them out, we get a smoothed version of them (low variance), still centered around the true density (low bias).
		- In order to get a good reduction in variance, we require that the models being aggregated be uncorrelated, so that they make “different errors”
		- ![img][image-11]
	- can be parallelized

- boosting

	- considers homogeneous weak learners, learns them sequentially in a very adaptative way (a base model depends on the previous ones) and combines them following a deterministic strategy
	- iteratively
	- each model in the sequence is fitted giving more importance to observations in the dataset that were badly handled by the previous models in the sequence. Intuitively, each new model focus its efforts on the most difficult observations to fit up to now, so that we obtain, at the end of the process, a strong learner with lower bias (even if we can notice that boosting can also have the effect of reducing variance). 
	- adaboost
		- Adaptive boosting updates the weights attached to each of the training dataset observations 
		- define our ensemble model as a weighted sum of L weak learners
		- ![img][image-12]
	- gradient boosting
		- gradient boosting updates the value of these observations.

- stacking

	- that often considers heterogeneous weak learners, learns them in parallel and combines them by training a meta-model to output a prediction based on the different weak models predictions
	- Very roughly, we can say that bagging will mainly focus at getting an ensemble model with less variance than its components whereas boosting and stacking will mainly try to produce strong models less biased than their components (even if variance can also be reduced).

## Evaluation
- accuracy
- precision
	- positive predictive value
	- out of all positive predictions, how much are actually positive 
	- precision = TP / (TP + FP)
- recall
	- true positive rate
	- out of all positive classes, how much we predicted positive
	- recall = TP / (TP + FN)
- F1 score
	- F1 = 2 Precision \* Recall / (Precision + Recall)
	- weighted average of the precision and recall 
- ROC curve
	- receiver operating characteristics 
	- balance between sensitivity (TP, y-axis) and specificity (fall-out or the probability for triggering a false alarm) (FP, x-axis)
	- contrast between TP and FP at various threshold for assigning observations to a given class
- AUC
	- area under the curve
	- represents the degree or measure of separability, how much the model is capable of distinguishing between classes
- sensitivity
- specificity
- TP: type I error
- FN: type II error
- t-test in the context of machine learning
- 
## Neural Networks
State the universal approximation theorem? What is the technique used to prove that?
What is a Borel measurable function?
Given the universal approximation theorem, why can’t a Multi Layer Perceptron (MLP) still reach an arbitrarily small positive error?

what is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?
In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?
What are the reasons for choosing a deep model as opposed to shallow model?
How Deep Learning tackles the curse of dimensionality?

How will you implement dropout during forward and backward pass?
What do you do if Neural network training loss/testing loss stays constant? (ask if there could be an error in your code, going deeper, going simpler…)
Why do RNNs have a tendency to suffer from exploding/vanishing gradient? How to prevent this? (Talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. Talk about gradient clipping, and discuss whether to clip the gradient element wise, or clip the norm of the gradient.)
Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?
Does using full batch means that the convergence is always better given unlimited power? (Beautiful explanation by Alex Seewald: https://www.quora.com/Is-full-batch-gradient-descent-with-unlimited-computer-power-always-better-than-mini-batch-gradient-descent)
What is the problem with sigmoid during backpropagation? (Very small, between 0.25 and zero.)

## Sequence Models
Write the equation describing a dynamical system. Can you unfold it? Now, can you use this to describe a RNN?
What determines the size of an unfolded graph?
What are the advantages of an unfolded graph?
What does the output of the hidden layer of a RNN at any arbitrary time t represent?
Are the output of hidden layers of RNNs lossless? If not, why?
RNNs are used for various tasks. From a RNNs point of view, what tasks are more demanding than others?
Discuss some examples of important design patterns of classical RNNs.
Write the equations for a classical RNN where hidden layer has recurrence. How would you define the loss in this case? What problems you might face while training it?
What is backpropagation through time?
Consider a RNN that has only output to hidden layer recurrence. What are its advantages or disadvantages compared to a RNN having only hidden to hidden recurrence?
What is Teacher forcing? Compare and contrast with BPTT.
What is the disadvantage of using a strict teacher forcing technique? How to solve this?
Explain the vanishing/exploding gradient phenomenon for recurrent neural networks.
Why don’t we see the vanishing/exploding gradient phenomenon in feedforward networks?
What is the key difference in architecture of LSTMs/GRUs compared to traditional RNNs?
What is the difference between LSTM and GRU?
Explain Gradient Clipping.
Adam and RMSProp adjust the size of gradients based on previously seen gradients. Do they inherently perform gradient clipping? If no, why?
Discuss RNNs in the context of Bayesian Machine Learning.
Can we do Batch Normalization in RNNs? If not, what is the alternative?
## Auto-encoders
What is an Autoencoder? What does it “auto-encode”?
What were Autoencoders traditionally used for? Why there has been a resurgence of Autoencoders for generative modeling?
What is recirculation?
What loss functions are used for Autoencoders?
What is a linear autoencoder? Can it be optimal (lowest training reconstruction error)? If yes, under what conditions?
What is the difference between Autoencoders and PCA?
What is the impact of the size of the hidden layer in Autoencoders?
What is an undercomplete Autoencoder? Why is it typically used for?
What is a linear Autoencoder? Discuss it’s equivalence with PCA. Which one is better in reconstruction?
What problems might a nonlinear undercomplete Autoencoder face?
What are overcomplete Autoencoders? What problems might they face? Does the scenario change for linear overcomplete autoencoders?
Discuss the importance of regularization in the context of Autoencoders.
Why does generative autoencoders not require regularization?
What are sparse autoencoders?
What is a denoising autoencoder? What are its advantages? How does it solve the overcomplete problem?
What is score matching? Discuss it’s connections to DAEs.
Are there any connections between Autoencoders and RBMs?
What is manifold learning? How are denoising and contractive autoencoders equipped to do manifold learning?
What is a contractive autoencoder? Discuss its advantages. How does it solve the overcomplete problem?
Why is a contractive autoencoder named so?
What are the practical issues with CAEs? How to tackle them?
What is a stacked autoencoder? What is a deep autoencoder? Compare and contrast.
Compare the reconstruction quality of a deep autoencoder vs. PCA.
What is predictive sparse decomposition?
Discuss some applications of Autoencoders.
## Representation Learning
What is representation learning? Why is it useful?
What is the relation between Representation Learning and Deep Learning?
What is one-shot and zero-shot learning (Google’s NMT)? Give examples.
What trade offs does representation learning have to consider?
What is greedy layer-wise unsupervised pretraining (GLUP)? Why greedy? Why layer-wise? Why unsupervised? Why pretraining?
What were/are the purposes of the above technique? (deep learning problem and initialization)
Why does unsupervised pretraining work?
When does unsupervised training work? Under which circumstances?
Why might unsupervised pretraining act as a regularizer?
What is the disadvantage of unsupervised pretraining compared to other forms of unsupervised learning?
How do you control the regularizing effect of unsupervised pretraining?
How to select the hyperparameters of each stage of GLUP?

## Monte Carlo Methods
What are deterministic algorithms?
What are Las vegas algorithms?
What are deterministic approximate algorithms?
What are Monte Carlo algorithms?

### Feature Selection

- some mindset
	- Curse of dimensionality
	- occam's razor: we like simple and explainable models
	- garbage in garbage out
- Filter based: specif some metric and filter features
	- Pearson correlation
		- check the absolute value of the Pearson's correlation between target and numerical features
		- keep the top n features
	- chi-square
		- calculate the chi-square metric between the target and numerical variables
		- only select the variable with the maximum chi-square values
- wrapper based: search problem
	- recursive feature elimination
		- select features recursively considering smaller and smaller set of features
		\- 
- embedded: some algorithm have built-in feature selection capabilities
	- lasso
	- Random forest
		\- 

### Handle Imbalanced dataset

- Can You Collect More Data?
	- More examples of minor classes
- Changing Your Performance Metric
	- accuracy -\> 
	- confusion matrix, precision, recall, F1 score (weighted avg of p & r)
	- Kappa: Classification accuracy normalized by the imbalance of the classes in the data.
	- ROC Curve
- Resample your data
	- add copies of instances from the under-represented class called over-sampling (or more formally sampling with replacement)
	- You can delete instances from the over-represented class, called under-sampling.
- Generate Synthetic Samples
	- randomly sample the attributes from instances in the minority class
		- You could sample them empirically within your dataset 
		- or you could use a method like Naive Bayes that can sample each attribute independently when run in reverse. You will have more and different data, but the non-linear relationships between the attributes may not be preserved.
	- systematic algorithms
		- SMOTE or the Synthetic Minority Over-sampling Technique
			- an oversampling method. It works by creating synthetic samples from the minor class instead of creating copies. The algorithm selects two or more similar instances (using a distance measure) and perturbing an instance one attribute at a time by a random amount within the difference to the neighboring instances.
	- Try different algorithm
		- That being said, decision trees often perform well on imbalanced datasets. The splitting rules that look at the class variable used in the creation of the trees, can force both classes to be addressed.
		- If in doubt, try a few popular decision tree algorithms like C4.5, C5.0, CART, and Random Forest.
	- Penalized Models
		- imposes an additional cost on the model for making classification mistakes on the minority class during training
		\- 
- Resource
	- [8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset][3]
	- [Python imbalanced-learn][4]

### Missing data

- delete
- fill with placeholder
- create another dummy column that represents if missing or not
	\- 

## Statistics


### MLE vs MAP

## Optimization
What is the difference between an optimization problem and a Machine Learning problem?
How can a learning problem be converted into an optimization problem?
What is empirical risk minimization? Why the term empirical? Why do we rarely use it in the context of deep learning?
Name some typical loss functions used for regression. Compare and contrast.
What is the 0–1 loss function? Why can’t the 0–1 loss function or classification error be used as a loss function for optimizing a deep neural network?  (Non-convex, gradient is either 0 or undefined. https://davidrosenberg.github.io/ml2015/docs/3a.loss-functions.pdf)

[1]:	https://en.wikipedia.org/wiki/Decision_tree_pruning
[2]:	https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
[3]:	https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
[4]:	https://github.com/scikit-learn-contrib/imbalanced-learn

[image-1]:	https://miro.medium.com/max/1052/1*55TfJMq5AkMKDg9VUa7ktA.jpeg
[image-2]:	https://miro.medium.com/max/2892/1*J2B_bcbd1-s1kpWOu_FZrg.png
[image-3]:	https://cdn-images-1.medium.com/max/1600/1*RqXFpiNGwdiKBWyLJc_E7g.png
[image-4]:	https://ml-cheatsheet.readthedocs.io/en/latest/_images/ng_cost_function_logistic.png
[image-5]:	https://ibb.co/WV4kvNc
[image-6]:	https://ibb.co/RTsXw5N
[image-7]:	https://www.datanovia.com/en/wp-content/uploads/dn-tutorials/004-cluster-validation/figures/015-determining-the-optimal-number-of-clusters-k-means-optimal-clusters-wss-silhouette-1.png
[image-8]:	https://cdn-images-1.medium.com/max/1600/0*AgmY9auxftS9BI73.png
[image-9]:	https://cdn-images-1.medium.com/max/1600/0*xTLQtW2XQY6P3mZf.png
[image-10]:	https://robjhyndman.com/files/cv1-1.png
[image-11]:	https://qph.fs.quoracdn.net/main-qimg-55c44d63831742ddd387541a428fcedf
[image-12]:	https://cdn-images-1.medium.com/max/1600/1*7wz2AIdH0pZSIUAxveLlIg@2x.png