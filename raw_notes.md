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






