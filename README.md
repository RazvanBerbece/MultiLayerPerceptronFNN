# FeedforwardNeuralNet
An abstract implementation of a Multilayered Perceptron (Multilayered Feedforward Neural Network). Provides a well-documented API which exposes a wrapper around the whole process (all the way from network config, to modelling, to training and predicting). Built in Python.

# Progress
[x] Neural Net Config (~~1 hidden layer~~, n hidden layers)

[x] Feedforward (~~sigmoid~~, ~~linear~~, ReLU (?))

[x] Training (~~GD~~, ~~Backpropagation~~, add check on MSE to stop further iterations when MSE goes down)

[x] Predicting

[ ] Stochastic Gradient Descent, Minibatch Gradient Descent 

# Example Models & API Usage (./models/...)
1. OR Gate (or.py) - a model that predicts the result of two binary inputs that go through an OR gate
2. Sum (sum.py) - a model that predicts the result of the sum between two numbers
