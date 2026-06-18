import numpy as np
from sklearn.datasets import load_diabetes

# creating a dataset using sklearn library 
X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import r2_score

print(f"R2 score of model is:  {r2_score(y_test, y_pred)}")

print(f"Betas(slop) matrics: {lr.coef_}")

print(f"intercept of our model plane: {lr.intercept_}")