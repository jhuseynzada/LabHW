This repository contains my solution (solution.m) file for the Intelligent Systems laboratory homework.
The task focuses on implementing a single-layer Perceptron classifier and comparing it with a Naive Bayes classifier using MATLAB.

The dataset is provided in Data.txt, and the goal is to classify two different types of objects based on two numerical features.

1. Perceptron Classifier

The first part of the assignment was to build a Perceptron from scratch, initialize parameters randomly, and train it using the rule:

Output function

y = 1, if (x₁·w₁ + x₂·w₂ + b > 0)

y = –1, otherwise

Parameter update rules:

w1(n+1) = w1(n) + η * e(n) * x1(n)
w2(n+1) = w2(n) + η * e(n) * x2(n)
b(n+1)  = b(n)  + η * e(n)


where
e(n) = d(n) – y(n) is the error and
0 < η < 1 is the learning rate.

The perceptron successfully converged, reaching 0 errors after a few epochs and achieving 100% accuracy on the training set. A decision boundary plot is also included for visualization.

2. Naive Bayes Classifier (Additional Task)

For the additional points, I implemented a Gaussian Naive Bayes classifier.
Each feature was modeled as an independent 1D Gaussian distribution per class.

The classifier computes:

Class priors

Feature means and variances

Log-likelihood of each class

Predicted class label using maximum posterior probability

The Naive Bayes classifier also achieved high accuracy on the provided dataset.

3. Files Included

IS_Lab1_solution.m
Complete MATLAB solution including:

Perceptron training

Decision boundary plot

Naive Bayes implementation

Accuracy comparison

Data.txt
Dataset of two-feature samples with class labels.

(Template files from the original lab, if included)

4. How to Run

Clone this repository:

git clone https://github.com/jhuseynzada/LabHW.git

Open MATLAB and set the folder as the working directory.

Run:

IS_Lab1_solution

The script will:

-Train the perceptron
-Display training progress
-Plot the decision boundary
-Train and evaluate the Naive Bayes classifier
