# NEURAL NETWORK FROM SCRATCH

See How Artificial Neural Network working from underneath and in 'low level' implementation.

This repo for those who already learn neural network, but want to know how actually this implemented in code without any Machine Learning framework.I use Python for simplycity and Numpy to use the powerful numpy array that simlpified matrix operations.

All module are not stored in class to simplified the step-by-step implementation, and to see what actually this function really does.

## What Does This Repo Covers :
1. Initializing parameters with random numbers
2. Forward Propagation
3. Simple Backward Propagation for Logistic Regression / Binary Classification
4. Updating parameters
5. Training Process

## Use of Each Module
- `main.py` is used to run a console program, which input data, and output a prediction.
- `train.py` used to train neural network model with the data the given. 
- All mathematics calculation and neural network functions are in `utils` folder.
    - `data.py` is store functions that related to input data, like generate data and load data
    - `math_calculation.py` is store functions to calculate things that used in neural network like compute z, z score norm, cost function, and others.
    - `neural_network.py` is store function of neural network that used in train and predict.

## General Process of Neural Network Trainig :
The Grand picture of how training in nn is:
1. Determine what spesification of network: number of layers, number of units for each layer, ouput layer's activation function
2. Generate matrix with random values for parameter W (Weight) and b (bias)
3. Normalize input data
4. Pass data to every unit in every layer, starts from first layer.
    - In every unit, data is used as input to basically line function (z = wx + b) per row, starts from first row
    - output from z is used as input to g activation function (in this repo g is sigmoid function)
    - return g as output
5. Output from first layer is forwarded as input for second layer, and so on until reach last layer / output layer
6. Compute cost function, a measure of how far our prediction yhat to real label y
7. Compute derivative for of cost function in respect for each parameter w (dJ/dw) and b (dJ/db) in corresponding cost value
8. Update parameters w with wi = wi - alpha * dJ/dWi where i is layer number, alpha is learning rate, and dJ/dWi is derivative of cost function
9. Run from 4 again with updated parameters until max iteration is reached (max iteration is determined by user)