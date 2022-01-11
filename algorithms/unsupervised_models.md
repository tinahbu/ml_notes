# Unsupervised Learning

What is distortion function? Is it convex or non-convex?
Tell me about the convergence of the distortion function.
Topic: EM algorithm
What is the Gaussian Mixture Model?
Describe the EM algorithm intuitively.
What are the two steps of the EM algorithm
Compare Gaussian Mixture Model and Gaussian Discriminant Analysis.

## K-means Clustering

- k-means clustering aims to partition $n$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 
- Steps
	- Initialize center points
		- Forgy method: randomly choose $k$ observations as centroids
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
		- The result may depend on the initial clusters. -> randomly initialize the group centers a few times, and then select the run that looks like it provided the best results.
- select $k$
    - the number of centroids you need in the dataset
    - Choose $k$ with the elbow method
        - run k-means clustering on the dataset for a range of values of *k* (say, *k* from 1 to 10 in the examples above), and for each value of *k* calculate the sum of squared errors (SSE) within the clusters. 
        - Then, plot a line chart of the SSE for each value of *k*. If the line chart looks like an arm, then the "elbow"/bend on the arm is the value of *k* that is the best.
        - The idea is that we want a small SSE, but that the SSE tends to decrease toward 0 as we increase *k* (the SSE is 0 when *k* is equal to the number of data points in the dataset, because then each data point is its own cluster, and there is no error between it and the center of its cluster). So our goal is to choose a small value of *k* that still has a low SSE, and the elbow usually represents where we start to have diminishing returns by increasing *k*.
        - ![Image result for elbow method][image-7]
        - However, the elbow method doesn't always work well; especially if the data is not very clustered. does not have a clear elbow. Instead, we see a fairly smooth curve, and it's unclear what is the best value of *k* to choose. 
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
	- performance is usually not as competitive as those of the other sophisticated clustering techniques because slight variations in the data could lead to high variance.

## Mean-Shift Clustering
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

## Frequent Itemset Mining	
- Apiori