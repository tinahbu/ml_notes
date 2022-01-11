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
	- learn the actual joint probability P(x, y)
- discrimative
	- learns the distinction between different categories
	- learns the conditional distribution P(y|x)
- Convex
	- smooth = infinitely derivable
	- convex = has a global optimum
- non-convex
## train/test split
- training set is used to train the model 
- the validation/test set is used to validate it on data it has never seen before
- why validation set
	- detect over-fitting and to assist in hyper-parameter search
	- so that you can evaluate the performance of your model for different combinations of hyperparameter values (e.g. by means of a grid search process) and keep the best trained model
- why test set
	- measure the performance of the model
	- compare different models against each other in an unbiased way, by basing your comparisons in data that were not use in any part of your training/hyperparameter selection process. 
	- You cannot compare based on the validation set, because that validation set was part of the fitting of your model. You used it to select the hyperparameter values!
	- unseen data
- `from sklearn.model_selection import train_test_split`

## cross validation
- ![][image-1]
```python
	from sklearn.cross_validation import cross_val_score, cross_val_pred
	# print cross validation scores
	scores = cross_val_score(model, df, y, cv=5)
	# make cross validation predictions
	predictions = cross_val_pred(model, df, y, cv=5)
```
- 3 different cross-validation techniques
	- simple k-folds
		```python
			from sklearn.model_selection import KFold
			kf = KFold(n_splits=5)
			kf.get_n_splits(X) 
			for train_index, test_index in kf.split(X):
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
		```
		- split our data into K parts, let’s use K=3 for example, part 1, part 2 and part 3. We then build three different models, each model is trained on two parts and tested on the third. Our first model is trained on part 1 and 2 and tested on part 3. Our second model is trained to on part 1 and part 3 and tested on part 2 and so on.
		- ![][image-2]
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
- benefit
	- hyperparameters tuning
		- the number of trees in Gradient Boosting classifier
		- hidden layer size or activation functions in a Neural Network
		- type of kernel in an SVM and many more
	- we’re able to use all our examples both for training and for testing while evaluating our learning algorithm on examples it has never seen before.
	- we can be more confident in our algorithm performance
		- if performance metrics very different across folds
			- algorithms inconsistent
			- or data inconsistent
				\- 
