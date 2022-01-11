# Supervised Learning

## Linear regression

- independent variables -> numeric outcome
- hypothesis:  $hΘ(x) = WX + B$
- least of squares: minimizing squared error of distance

## Logistic regression

- binary classification
- hypothesis: $Z = WX + B$, $hΘ(x) = sigmoid (Z)$
	- linear classifier (or single layer perceptron)
- sigmoid (logistic) function: $1 / (1 + e^{-z})$
	- $(-∞,+∞)$ -> $(0, 1)$
	- predictions -> probabilities of classes
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

## Decision Tree

- regression & classification
- learns decision rules with a tree representation
- tree nodes corresponds to attributes
- leaf nodes corresponds to a prediction
- the higher the node, the more important its attribute
- criteria
	- gini
	- information gain
	- entropy
- regularization -> pruning
	- remove branches that have weak predictive power
	- to reduce model complexity and predictive accuracy
	- bottom-up or top-down
	- reduced error pruning: Starting at the leaves, each node is replaced with its most popular class. If the prediction accuracy is not affected then the change is kept. While somewhat naive, reduced error pruning has the advantage of simplicity and speed.
	- cost complexity pruning: [https://en.wikipedia.org/wiki/Decision\_tree\_pruning][1]

## SVM - Support Vector Machine

- classification
- Produces nonlinear boundaries by constructing a linear boundary in a large, transformed version of the feature space
- Maximum margin classifier
	- fit the maximum-margin hyperplane in a transformed feature space
	- large margin for all classes, best separates the classes
    - margin: the distance between the hyperplane and the closest class point
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

## Naive Bayes

- naive: independence between features
	- Bayes Theorem
		- $P(c|X) = P(X|c) P(c) / P(X)$ where $c$ is the class and $X$ is the attibutes
		- $P(A|B) = P(B|A)P(A)/P(B)$
		- posterior = likelihood * class prior / predictor prior
- binary classification: popular for spam filtering

### KNN (k-nearest Neighbors)

- classifies new cases by a majority vote of its $k$ neighbors
- sometimes also regression
- lazy learner
	- does not have any learning involved, i.e., there are no parameters we can tune to make the performance better. Or we are not trying to optimize an objective function from the training data set
	- all computation is deferred until classification
	- non-parametric method used for classification and regression
- Steps
	- initialize $k$ to your chosen number of neighbors
	- for each example in the data
		- compute distance between the query example and the current example from the data
		- add the distance and the index of the example to an ordered collection
	- sort the ordered collection of distances from smallest to largest (ascending)
	- pick the first $k$ entries, get the labels/values of the $k$ entries
	- or assign weight to the contributions of all the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of $1/d$, where $d$ is the distance to the neighbor
	- if regression, return the mean; if classification return the most popular vote
- distance function
	- continuous
		- Euclidean: straight line distance between two points
		- Manhattan: sum of absolute differences
		- Minkowski
	- categorical
		- Hamming
	- requires normalization
- Cons
	- expensive O(N^2)

## Random Forest

- classification & regression
- bagging method where deep tree, fitted on bootstrap samples, are combined to produce an output with lower variance. each tree votes and makes final decision based on majority vote
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


