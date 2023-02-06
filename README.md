# Neural Network for taxi fare prediction

## Introduction
This project aims to build a neural network to predict taxi fares based on the given dataset. The project consists of several parts to experiment with different hyperparameters to improve the performance of the model.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Numpy
- Pandas
- Matplotlib
- Seaborn

## Parts
1. Part A: Create a baseline Neural network with 2 hidden layers, each with 16 and 8 neurons respectively, Sigmoid activation, and batch size of 128 for Gradient Descent.
2. Part B: Experiment with number of layers and neurons per layer.
3. Part C: Experiment with activation functions.
4. Part D: Experiment with regularization techniques: Early stopping and Dropout rate. 
5. Part E: Experiment with at least 2 optimization methods. 

## Evaluation
The results of each part will be tabulated with 95% confidence intervals of each of the 3 metrics (MSE, MAE, MAPE) based on at least 5 experiments on validation. The best architecture and hyperparameters will be reported and used to generate predictions for the test set.
## Files

- `Taxi-price-prediction.ipynb`: Jupyter Notebook containing the code for training and evaluating the Neural network
- `data`: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
## Usage
1. Clone the repository ---> git clone https://github.com/[username]/taxi_fare_prediction.git
2. Install the required packages
3. Run the Jupyter notebook ---> jupyter notebook taxi_fare_prediction.ipynb

## Conclusion
In this project, we experimented with different hyperparameters of a neural network to predict taxi fares based on the given dataset. The best architecture and hyperparameters were reported based on the evaluation metrics.

