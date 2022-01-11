# Data

## Data Collection

- online
	- user interaction with system
		- e.g. get training data for linkedin post (click thru rate)
			- ranks post in chronological order, collect click/non-click data. user lose attention after a while + data sparsity problem (some activities like friends changed jobs are much rarer) 
			- random serve posts -> bad user experience, still data sparsity problem
			- use a feed ranking algorithm, then random serve the top feeds. allow randomness & allows models to learn and explore more activities
	- design systems to collect data ADD PICTURE
- offline
	- crowdsourcing, open source datasets
	- trained human labelers: expensive
	- when users are not interacting with the system in a way that will generate labeled data
	- manually expand/enhance ADD PICTURE
	- data expansion using GANs (generative adversarial networks) 

## Data Storage

- store data in column-oriented format
	- partitioned by time to avoid scanning through the whole dataset. speed up queries compared to csv by reducing the data scanned (also saving cost)
	- parquet
	- ocr
- tfrecord for Tensorflow

## Data Cleaning

### Filtering

- avoid bias

### Imbalance class distribution

- use cases
	- fraud detection
	- click prediction
	- spam detection
- class weights in loss function
	- penalize more of the majority class 
	- loss_function = $-w_0 * ylog(p) - w_1*(1-y)*log(1-p)$
- naive resampling
	- resample the majority class at a certain rate to reduce the imbalance in the training set
	- also called negative down-sampling
	- keep validation & test data intact with no sampling
- synthetic sampling: SMOTE
	- synthetic minority over-sampling technique
	- synthesizing elements for the minority class, based on those that already exist
	- randomly picking a point from the minority class and compute the k-nearest neighbors for that point, then add the synthetic points


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