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

### Bayesian Machine Learning

What are the differences between “Bayesian” and “Freqentist” approach for Machine Learning?
Compare and contrast maximum likelihood and maximum a posteriori estimation.
How does Bayesian methods do automatic feature selection?
What do you mean by Bayesian regularization?
When will you use Bayesian methods instead of Frequentist methods?

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


## NLP

- language understanding, speech recognition, entity recognition, language generation, semantic understanding
- word embeddings
	- word2vec
	- BERT
	- ELMO

# INFERENCE

- use a trained model to make a prediction
    - SLA
        - throughput & latency: e.g. recommend 100 videos to user under 100 ms
- multi-stages so system can scale
    - e.g. for video recommendation, generate candidate -> rank
    - generate candidates with matrix factorization: decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices
    - In practice, for large scale system (Facebook, Google), we don’t use Collaborative Filtering but leverage Inverted Index (commonly used in Lucene, Elastic Search) 
    - rank videos based on the view likelihood
        - feature engineering
            - watched video ids -> video embedding
            - historical search query -> text embedding
            - location -> geolocation embedding
            - user associated features: age, gender -> normalization/standardization
            - previous impression -> normalization/standardization
            - time related features -> month, week_of_year, holiday, day_of_week, hour_of_day
        - model
            - logistic regression: can be trained distributedly in spark
            - neural network with fully connected layers with relu activation function and sigmoid function at final layer


video recommendation system.png 
video recommendation pipeline.png


##### user/video db: metadata about user & videos

###### use this for collaborative filtering

##### 

##### user historical recommendations:past recommendations

##### user watched history: videos a user have watched over time

##### search query db: user's historical queries

##### 

###### use these for ranking

##### resampling data: too many negatives (recommended but not watched), downsample them

##### but keep the validation & test sets intact to have accurate model performance estimates

## imbalance workload
- split workloads onto multiple inference servers
- similar as load balancers
- sometimes called an aggregator service 
    - clients send requests to aggregator service
    - splits the workload and sends to workers
    - pick workers based on
        - work load
        - round robin
        - request parameters
    - wait for response from workers
    - forward response to client









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