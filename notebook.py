# %% markdown
# # Implementing Decision Tree Algorithm from scratch
#
# The purpose of the tutorial is to implement step-by-step a Decision Tree
# algorithm from scratch using both ID3 (Iterative Dichotomiser 3) and CART
# (Classification And Regression Tree) algorithms.
# As we code along, we will dive more in deepth in each algorithms.
# Most of the code in made in the DecisionTree.py file at the root of the
# project.
# In order to test our models, we will use the Titanic dataset available [here](https://www.kaggle.com/c/titanic/data)

# %%
# Auto reload external librairies.
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from DecisionTree import DecisionTree

titanic_df = pd.read_csv('titanic.csv', sep='\t')
titanic_df.head()

# %%
from sklearn.model_selection import train_test_split

X = titanic_df.iloc[:, 2:]
y = titanic_df.iloc[:, 1]

# %%
# Convert gender tosupport DecisionTreeClassifier male = 0, female = 1
X['Sex'] = [int(sex == 'female') for sex in X['Sex']]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)
X_y_train = pd.concat([X_train, y_train], axis=1)

# %%
dt = DecisionTree()

# Defines features used in classification
features = ['Sex', 'Pclass', 'SibSp']
dt.fit(X_train, y_train, features=['Pclass', 'SibSp', 'Sex'])

# %%
y_hat_train = dt.predict(X_train)
y_hat_test = dt.predict(X_test)

# %%
print('-------- Predictions with Train set ---------')
train_scratch_acc = dt.accuracy(y_train, y_hat_train)
print(train_scratch_acc)
print('-------- Predictions with Test set ---------')
test_scratch_acc = dt.accuracy(y_test, y_hat_test)
print(test_scratch_acc)

# %%
# Compare with sklearn built-in classifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train[features], y_train)

y_hat_train_cls = classifier.predict(X_train[features])
y_hat_test_cls = classifier.predict(X_test[features])

print('-------- Predictions with sklearn Train set ---------')
# print(confusion_matrix(y_train, y_hat_train_cls))
# print(classification_report(y_train, y_hat_train_cls))
train_acc = dt.accuracy(y_train, y_hat_train_cls)
print(train_skl_acc)
print('-------- Predictions with sklearn Test set ---------')
# print(confusion_matrix(y_test, y_hat_test_cls))
# print(classification_report(y_test, y_hat_test_cls))
test_acc = dt.accuracy(y_test, y_hat_test_cls)
print(test_skl_acc)
