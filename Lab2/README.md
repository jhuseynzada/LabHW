# Multi-Layer Perceptron – Function and Surface Approximation (MATLAB)

This project contains my implementation of a Multi-Layer Perceptron (MLP) trained from scratch using backpropagation. The goal is to approximate a nonlinear 1D function and then extend the same approach to a 2D surface. Everything is written manually in MATLAB without using the Neural Network Toolbox.

## Task 1 – 1D Function Approximation

In the first part, I generate 20 input values between 0 and 1 and compute the target values using the function provided in the assignment.  
The MLP architecture used here is:

- 1 input neuron  
- 1 hidden layer (tanh activation)  
- 1 linear output neuron  

The training is done with online backpropagation, meaning weights are updated after each training sample.  
After training, the script plots both the original function and the MLP’s approximation so the comparison is clear.

## Task 2 – 2D Surface Approximation (Additional Task)

The second part extends the network to two inputs.  
A grid of (x1, x2) values is created, and a smooth surface defined by a sine–cosine function is used as the target. The MLP architecture becomes:

- 2 input neurons  
- 1 hidden layer (tanh activation)  
- 1 linear output neuron  

The same backpropagation logic is used here, just adapted for two inputs.  
After training, the script visualizes:

- the true target surface  
- the MLP-approximated surface  
- the error surface  

This shows how well the model fits the underlying function.

## Summary

The code demonstrates the core mechanisms of training a neural network: the forward pass, gradient computation, and weight updates.  
Both the 1D and 2D tasks show how an MLP can learn nonlinear relationships when trained properly with backpropagation.
