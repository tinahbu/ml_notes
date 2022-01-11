# Statistics
## Estimation
- estimate unknown parameter, given some data
- y = y(x) + epsilon
	- epsilon: the part of y that is not predictable from x
## estimators
- Maximum likelihood estimator (MLE)
	- argmax P(Y | X, W)
	- parameters that maximizes the probability of obtaining the observed data
- Maximum a Posteriori (MAP)
	- argmax P(W | X, Y)
	- parameters that maximizes posterior probability
	- leverage prior information
	- reduces variance, increase bias
- MAP is same as MLE when
	- infinite amount of data
	- uniform prior (infinitely weak prior belief)
- MLE used with regularization, MAP uses prior like a regularization
## i.i.d
