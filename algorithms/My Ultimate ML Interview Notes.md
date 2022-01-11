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

### Bayesian Machine Learning

What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
Compare and contrast maximum likelihood and maximum a posteriori estimation.
How does Bayesian methods do automatic feature selection?
What do you mean by Bayesian regularization?
When will you use Bayesian methods instead of Frequentist methods?

### Reinforcement Learning

- Markov Decision Process


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