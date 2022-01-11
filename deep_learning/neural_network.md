# Neural Networks

State the universal approximation theorem? What is the technique used to prove that?

What is a Borel measurable function?

Given the universal approximation theorem, why can’t a Multi Layer Perceptron (MLP) still reach an arbitrarily small positive error?


what is the mathematical motivation of Deep Learning as opposed to standard Machine Learning techniques?

In standard Machine Learning vs. Deep Learning, how is the order of number of samples related to the order of regions that can be recognized in the function space?

What are the reasons for choosing a deep model as opposed to shallow model?

- How Deep Learning tackles the curse of dimensionality?
    - The curse of dimensionality normally comes about because in data there are relevant and too many irrelevant (noise) features.
    - Manifold Hypothesis
        - high dimensional data actually sits on a lower dimensional manifold embedded in higher dimensional space
        - neural networks is good at finding low dimensional features that are not apparent in the high dimensional representation
    - Sparse Coding
        - This is the occurrence when the data flows through the different neurons in a network. Each neuron in the Neural network has it’s own activation functions. When each neuron fires on it’s activation it is causing sparse coding.
        - In terms of neural networks there are some number of neurons that when fired can be combined with the firing of other neurons to combine to output the correct answer no matter the dimensionality of the inputs.

- regularization
    - dropout

What do you do if Neural network training loss/testing loss stays constant? (ask if there could be an error in your code, going deeper, going simpler…)

Why do RNNs have a tendency to suffer from exploding/vanishing gradient? How to prevent this? (Talk about LSTM cell which helps the gradient from vanishing, but make sure you know why it does so. Talk about gradient clipping, and discuss whether to clip the gradient element wise, or clip the norm of the gradient.)

Do you know GAN, VAE, and memory augmented neural network? Can you talk about it?

Does using full batch means that the convergence is always better given unlimited power? (Beautiful explanation by Alex Seewald: https://www.quora.com/Is-full-batch-gradient-descent-with-unlimited-computer-power-always-better-than-mini-batch-gradient-descent)

What is the problem with sigmoid during backpropagation? (Very small, between 0.25 and zero.)
