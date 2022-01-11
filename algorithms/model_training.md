# Model Training Considerations

## Model Selection

- Considerations
	- model performance
	- complexity, maintainability, and available resources
- Probabilistic methods
	- score models both on their performance and complexity 
	- Akaike Information Criterion (AIC)
	- Bayesian Information Criterion (BIC)	
		- ACI and BIC are two ways of scoring a model based on its log-likelihood and complexity.
	- Minimum Description Length (MDL)
		- provides another scoring method from information theory that can be shown to be equivalent to BIC
	- Structural Risk Minimization (SRM)
- Resampling methods
	- Random train/test splits (hold out)
		- training set is used to train the model 
		- the validation/test set is used to validate it on data it has never seen before
		- why training set
			- fit model parameters
		- why validation set
			- detect over-fitting and to assist in hyper-parameter search
			- so that you can evaluate the performance of your model for different combinations of hyperparameter values (e.g. by means of a grid search process) and keep the best trained model
		- why test set
			- measure the performance of the model
			- compare different models against each other in an unbiased way, by basing your comparisons in data that were not use in any part of your training/hyperparameter selection process. 
			- You cannot compare based on the validation set, because that validation set was part of the fitting of your model. You used it to select the hyperparameter values!
			- unseen data
		 `from sklearn.model_selection import train_test_split`
		 - use hold out instead of cross validation when data set is really big or need to save time
	- Cross-Validation
		```python
		from sklearn.cross_validation import cross_val_score, cross_val_pred
		# print cross validation scores
		scores = cross_val_score(model, df, y, cv=5)
		# make cross validation predictions
		predictions = cross_val_pred(model, df, y, cv=5)
		```
		- k-folds
			![k folds cv](https://miro.medium.com/proxy/1*NyvaFiG_jXcGgOaouumYJQ.jpeg)
			```python
			from sklearn.model_selection import KFold
			kf = KFold(n_splits=5)
			kf.get_n_splits(X) 
			for train_index, test_index in kf.split(X):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
			```
			- split our data into K parts, let’s use K=3 for example, part 1, part 2 and part 3. We then build three different models, each model is trained on two parts and tested on the third. Our first model is trained on part 1 and 2 and tested on part 3. Our second model is trained to on part 1 and part 3 and tested on part 2 and so on.
			- each example appears in a test set only once
		- Leave One Out
			```python
			from sklearn.model_selection import LeaveOneOut
			loo = LeaveOneOut()
			loo.get_n_splits(X)
			```
			- For each instance in our dataset, we build a model using all other instances and then test it on the selected instance.
		- k-fold vs leave one out
			- the more the folds, the longer and more memory computation takes
			- the fewer the folds, the more bias there is
			- small set: leave one out
			- bigger set: k=3
		- Stratified Cross Validation
			- keep the same proportion of different classes in each fold
		- benefit of cross-validation
			- hyperparameters tuning
				- the number of trees in Gradient Boosting classifier
				- hidden layer size or activation functions in a Neural Network
				- type of kernel in an SVM and many more
			- we’re able to use all our examples both for training and for testing while evaluating our learning algorithm on examples it has never seen before.
			- we can be more confident in our algorithm performance
				- if performance metrics very different across folds -> algorithms inconsistent or data inconsistent
	- Bootstrap
		- a resampling technique used to estimate statistics on a population by sampling a dataset with replacement
		-  a data point in a drawn sample can reappear in future drawn samples as well
		- bootstrap aggregating (also called bagging). It helps in avoiding overfitting and improves the stability of machine learning algorithms.
		- In bagging, a certain number of equally sized subsets of a dataset are extracted with replacement. Then, a machine learning algorithm is applied to each of these subsets and the outputs are ensembled as I have illustrated below:
- time series data
	- cross-validation
	- validation set needs to come chronologically after the training subset
	- In this procedure, there is a series of test sets, each consisting of a single observation. The corresponding training set consists only of observations that occurred prior to the observation that forms the test set. Thus, no future observations can be used in constructing the forecast. The following diagram illustrates the series of training and test sets, where the blue observations form the training sets, and the red observations form the test sets.
	- The forecast accuracy is computed by averaging over the test sets. This procedure is sometimes known as “evaluation on a rolling forecasting origin” because the “origin” at which the forecast is based rolls forward in time.
	- forward chaining
		- multiple train test sets
		- test sets only have 1 observation
		- corresponding training set consists of all the observations that occurred prior to the test observation (no future observation)
		- ![Image result for time series cross validation][image-10]
		- The forecast accuracy is computed by averaging over the test sets


## Model Evaluation

- what to evaluate
	- tune hyperparameters 
	- how big training set needs to be
	- how frequent to retrain the model
- model performance metrics
	- accuracy
	- confusion matrix
		![confusion matrix](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/06/Basic-Confusion-matrix.png)
	- precision
		- positive predictive value
		- out of all positive predictions, how much are actually positive 
		- precision = TP / (TP + FP)
		- TP: type I error, FN: type II error
	- recall/sensitivity
		- true positive rate
		- out of all positive classes, how much we predicted positive
		- recall = TP / (TP + FN)
	- F1 score
		- F1 = 2 Precision * Recall / (Precision + Recall)
		- weighted average of the precision and recall 
	- specificity
		- specificity = TN / (TN + FP)
		- out of all the negative classes, how much we predicted negative
	- TPR (true positive rate) = recall = TP / (TP + FN) 
	- FPR (false positive rate) = 1 - specificity = FP / (TN + FP)
	- ROC (receiver operating characteristics)
		- binary classification
		- balance between sensitivity (TPR, y-axis) and specificity (fall-out or the probability for triggering a false alarm) (FPR, x-axis)
		- contrast between TP and FP at various threshold for assigning observations to a given class
		- a probability curve that plots the TPR against FPR at various classification threshold 0and essentially separates the 'signal' from the 'noise' 
	- AUC (area under the curve)
		- represents the degree or measure of separability, how much the model is capable of distinguishing between classes
		-  used as a summary of the ROC curve
		- When AUC = 1, then the classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly. If, however, the AUC had been 0, then the classifier would be predicting all Negatives as Positives, and all Positives as Negatives. When AUC=0.5, then the classifier is not able to distinguish between Positive and Negative class points. Meaning either the classifier is predicting random class or constant class for all the data points.
		- AUC is desirable for the following two reasons:
			- AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
			- AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.
		- However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:
			- Scale invariance is not always desirable. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that
			- Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

	- t-test in the context of machine learning


## Choose the right loss function

- binary classification
	- cross entropy
		- $-(ylog(p)+(1-y)log(1-p))$
		- only applicable for binary classification & sensitive to background click through rate
	- normalized cross entropy
		- predictive logloss / cross emtropy of background click through rate => insensitive to background CTR
	- AUC
	- logloss: $-p1log(p1)-p2log(p2)-p3log(p3)-...$
	- in the click through rate (CTR) prediction, facebook uses normalized cross entropy aka logloss to make the loss less sensitive to the background conversion rate
- forecast
	- mean absolute percentage error (MAPE)
    	- is your target value skew?
		- MAPE pic
	- symmetric absolute percentage error (SMAPE)
    	- not symmetric, treats under and over forecast differently
		- SMAPE pic
	- uber uses RNN, gradient boosting trees, and support vector regressors for various problems including marketplace forecasting, hardware capacity planning, and marketing
	- doordash uses quantile loss to forecast food delivery demand
		- quantile loss pic

## Requirements

- handle large volume of data
- at low costs


## Re-training

- reason
	- because data distribution is not stationary
	- temporal changes: new viral video to recommend
	- keep the model fresh to sustain performance 
	- avoid showing repetitive feed to user
	- especially in AdTech & recommendation/personalization use cases. important to capture changes in user's behavior and trending topics
	- how to adapt to user behavior changing over time?
		- multi-arm bandit
		- bayesian logistic regression model to update prior data
		- use different loss functions to be less sensitive with click through rates
- requirements
	- fast training pipeline that scales well with big data
	- balance between model complexity & training time
- algorithm: bayesian logistic regression
- design a scheduler to retrain model on a regular basis
	- airflow
		- pros: good GUI; strong support community; independent scheduler 
		- cons: less flexibility; not easy to manage massive pipelines
	- luigi
		- pros: has a lot of libraries (hive, pig, google big query, redshift)
		- cons: not very scalable (tight with cron); not easy to create/test tasks
- save model to S3


#### in the click through rate (CTR) prediction, facebook uses normalized cross entropy aka logloss to make the loss less sensitive to the background conversion rate

### forecast

#### mean absolute percentage error (MAPE)

##### is your target value skew?

##### 

#### symmetric absolute percentage error (SMAPE)

##### not symmetric, treats under and over forecast differently

##### 

#### uber uses RNN, gradient boosting trees, and support vector regressors for various problems including marketplace forecasting, hardware capacity planning, and marketing

#### doordash uses quantile loss to forecast food delivery demand



# Optimization

## Optimizers
- updates weights and biases to reduce the error
- gradient descent
	- keep updating the model parameters to move closer to the values that results in smaller cost
	- partial derivative of cost function with respect to each model parameter
	- learning rate: alpha
	- batch
		- use all training instances to update the model parameters in each iteration
		- independent, parallizable
	- mini-batch
		- use smaller batch size
		- check how much parallization your batch has (GPU)
	- stochastic gradient descent (SGD)
		- update parameters using 1 training instance
			- converge more quickly
	- Adagrad
		- adapts learning rate specifically to individual features
			- sparse dataset
			- diminishing learning rate problem
	- RMSprop
		- instead of letting all the gradients accumulate for momentum, only calculates gradient in a fixed window
	- Adam
		- adaptive moment estimation
		- adding fractions of past gradients to calculate current gradients

## Model Categories

- generative
	- learn categories of data 
	- learn the actual joint probability $P(X, y)$
- discrimative
	- learns the distinction between different categories
	- learns the conditional distribution $P(y|X)$
- Convex
	- smooth = infinitely derivable
	- convex = has a global optimum
- non-convex


# OTHER MODEL CONSIDERATIONS

## Testing

## Debugging

- is feature's distribution different in test vs prod?
- change in seasonality? (training data is out of date)
- is feature engineering done the same offline & online?
- did our model over-fit?
	- use validation set for final model quality measurement
- did our model under-fit?
	- more complex model
	- more feature interactions
- can we add more features?
	- e.g., consider a scenario where a movie actually liked by the user was ranked very low by our recommendation system. On debugging, we figure out that the user has previously watched two movies by the same actor, so adding a feature on previous ratings by the user for this movie actor can help our model perform better in this case.
- do we not have enough training data?
	- underperforming in some scenarios -> increase examples accordingly


## Transfer Learning

- apply pre-learned model to new task -> higher accuracy & less time
- many problems share common sub-problems
	- e.g. understanding texts -> recommender system; ads; search
- limited training data. and some pre-trained model don't need labeled training data (word2vec)
- 2 ways to use transfer learning
	- Extract features from useful layers: Keep the initial layers of the pre-trained model and remove the final layers. Add the new layer to the remaining chunk and train them for final classification.
	- Fine-tuning: Change or tune the existing parameters in a pre-trained network, i.e., optimizing the model parameters during training for the supervised prediction task. 
		- say with ImageNet, first few layers are detecting edges, or colors, can keep. then the next few layers are detecting shapes. freeze the weight of most of the starting layers of the pre-trained model and fine-tune only the end layers
		- the more training data you have, the more layers you can fine tune

## iterative model improvement

## Serving Logic w/ Multiple Models

- change logic in serving models
    - e.g. in ad prediction systems, depending on the type of ad candidates (ios device vs android), round to different models for scoring
- exploration-exploitation tradeoff
	- give user too many new ads -> low conversions give user too few ads -> not enough exploration
	- over-exploits historic data -> user dont view new videos too much fresh new content -> may not be very relevant 
- what if model is under-explored?
	- introduce randomization, 2% of requests can get random candidates the rest gets sorted candidates
- thompson sampling: decide at time t, which action to take based on the reward