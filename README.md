# Implementing Decision Tree Algorithm from scratch

The purpose of the tutorial is to implement step-by-step a Decision Tree
algorithm from scratch using both ID3 (Iterative Dichotomiser 3) and CART
(Classification And Regression Tree) algorithms.

As we code along, we will dive more in depth in each algorithms.
Most of the code in made in the DecisionTree.py file at the root of the
project.

# Setup
In order to test our models, we will use the Titanic dataset available [here](https://www.kaggle.com/c/titanic/data)



# Theory

## Advantages

* Easy to interpret and visualise
* Works on non-linear patterns
* No need to preprocess/normalise the data prior running models
* No assumption needed regarding features distribution

## Disadvantages

* Sensitive to data noise. Easy to overfit the model
* Sensitive to imbalance dataset
