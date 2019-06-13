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
For more detail regarding the Theory behind the Decision tree, I have listed bellow
a serie of useful links explaining how it works and the variants:
* https://www.python-course.eu/Decision_Trees.php
* https://www.datacamp.com/community/tutorials/decision-tree-classification-python
* Book from Stephen Marsland https://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

# PRO vs CONS

## Advantages

* Easy to interpret and visualise
* Works on non-linear patterns
* No need to preprocess/normalise the data prior running models
* No assumption needed regarding features distribution

## Disadvantages

* Sensitive to data noise. Easy to overfit the model
* Sensitive to imbalance dataset
