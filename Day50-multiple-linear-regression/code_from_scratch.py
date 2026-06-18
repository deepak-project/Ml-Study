import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# add one extra column of 1 in our matrics because we want X matrics
X_train = np.insert(X_train,0,1,axis=1)

class MeraLR:

    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X_train, y_train):
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept = betas[0]
        self.coef = betas[1:]


    def predict(self, X_test):
        y_pred = self.intercept + np.dot(X_test, self.coef)
        return y_pred
    

model = MeraLR()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"predict value {y_pred}")

from sklearn.metrics import r2_score

print(f"R2_score {r2_score(y_test, y_pred)}")
        
print(f"Betas(slop) matrics: {model.coef}")

print(f"intercept of our model plane: {model.intercept}")