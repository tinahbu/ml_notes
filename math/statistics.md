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


## Confidence Interval
What is population mean and sample mean?
What is population standard deviation and sample standard deviation?
Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.?
What is the formula for calculating the s.d. of the sample mean?
What is confidence interval?
What is standard error?
