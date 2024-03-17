# Weights-Biases

# Neural Network from Scratch for Fashion MNIST

This project implements a neural network from scratch using Python and NumPy to classify images from the Fashion MNIST dataset.

## Introduction

The Fashion MNIST dataset is a collection of 28x28 grayscale images of fashion items, such as shirts, pants, shoes, and more. The goal of this project is to build a neural network model that can accurately classify these images into their respective categories.

## Features

- Supports customizable hyperparameters for training the neural network.
- Allows selection of different activation functions, loss functions, optimizers, and more.
- Provides options for configuring the number of layers, hidden units, batch size, and learning rate.

## Files

- gradient_descent.ipynb: Notebook for running hyperparameter sweeps with various configurations, implementing optimizers, and training model functions.
- train.py: Python script for model training, including command-line interface support using ArgParse.
- requirements.txt: File listing all the dependencies and packages required to run the project. It includes libraries needed for training, visualization, and hyperparameter tuning.
- README.md: Project's README file containing detailed information about the project, setup instructions, usage guidelines, and contribution guidelines.

## Instructions

1. **Clone the Repository:**
   - Clone the repository using Git:
     ```
     git clone https://github.com/pratik-kadlak/Weights-Biases.git
     ```

2. **Install Dependencies:**
   - Install the required dependencies listed in `requirements.txt` using pip:
     ```
     pip install -r requirements.txt
     ```

3. **Run the gradient_descent.ipynb Notebook for Hyperparameter Sweeps:**
   - Open and run the `optimizers_wandb.ipynb` notebook to perform hyperparameter sweeps, implement optimizers, and train the model.

4. **Train the Model using train.py and Command Line Interface:**
   - Run the `train.py` script with desired hyperparameters. For example:
     ```
     python train.py --learning_rate 0.01 --batch_size 64 --epochs 20 --optimizer adam
     ```
   Replace the hyperparameters (`--learning_rate`, `--batch_size`, `--epochs`, `--optimizer`, etc.) with values of your choice.

Here is the list of hyperparameter that you can pass to train the model.



Arguments to be supported
Name	Default Value	Description
| Argument           | Description                                                    | Default Value |
|--------------------|----------------------------------------------------------------|---------------|
| -wp, --wandb_project | Project name used to track experiments in Weights & Biases dashboard | DL_Assignment_1 |
| -we, --wandb_entity  | Wandb Entity used to track experiments in the Weights & Biases dashboard | DL_Assignment_1 |
| -d, --dataset       | Dataset used for training. Choices: ["mnist", "fashion_mnist"] | fashion_mnist |
| -e, --epochs        | Number of epochs to train neural network                        | 10            |
| -b, --batch_size    | Batch size used to train neural network                         | 16            |
| -l, --loss          | Loss function used. Choices: ["mean_squared_error", "cross_entropy"] | cross_entropy |
| -o, --optimizer     | Optimizer used for training. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | adam |
| -lr, --learning_rate | Learning rate used to optimize model parameters                 | 0.0001       |
| -m, --momentum      | Momentum used by momentum and nag optimizers                    | 0.9           |
| -beta, --beta       | Beta used by rmsprop optimizer                                  | 0.9           |
| -beta1, --beta1     | Beta1 used by adam and nadam optimizers                         | 0.9           |
| -beta2, --beta2     | Beta2 used by adam and nadam optimizers                         | 0.999         |
| -eps, --epsilon     | Epsilon used by optimizers                                      | 0.000001      |
| -w_d, --weight_decay | Weight decay used by optimizers                                 | 0.0005        |
| -w_i, --weight_init | Weight initialization method. Choices: ["random", "Xavier"]      | Xavier        |
| -nhl, --num_layers  | Number of hidden layers used in feedforward neural network      | 5             |
| -sz, --hidden_size  | Number of hidden neurons in a feedforward layer                 | 128            |
| -a, --activation    | Activation function used. Choices: ["identity", "sigmoid", "tanh", "ReLU"] | tanh |

## Report Link
https://wandb.ai/space_monkeys/DL_Assignment_1/reports/CS6910-Assignment-1--Vmlldzo3MTI2NDU2


