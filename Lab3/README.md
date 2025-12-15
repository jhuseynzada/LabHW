IS-Lab3 – RBF Network Training
Description

This laboratory work demonstrates the implementation and training of a Radial Basis Function (RBF) neural network for a function approximation task. The network consists of a single input, a single output, and two Gaussian radial basis functions in the hidden layer.

Objective

The main objective is to learn how to train the output layer of an RBF network using the perceptron (LMS) algorithm and to analyze how different network and learning parameters affect approximation accuracy.

Implementation

Input data consists of 20 samples generated in the interval 
[
0.1
,
1
]
[0.1,1].

Target values are computed using a nonlinear sinusoidal function.

Two Gaussian RBFs are used as nonlinear basis functions.

Centers and radii are manually selected for the base task.

Output layer weights are trained using the LMS (delta rule).

Additional Task

An alternative training approach is implemented where RBF centers and radii are also updated during training using gradient descent, allowing the network to adapt its basis functions and improve approximation performance.

Result

The trained RBF network successfully approximates the target function. Adaptive training of RBF parameters generally achieves lower mean squared error compared to fixed-parameter training.
