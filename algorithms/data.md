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
