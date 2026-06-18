import numpy as np
import pandas as pd

df = pd.read_csv('E:\ML Learn\ML Project\Creating-SLR\placement.csv')
 

X = df.iloc[:, 0:1] # seprating the input and output columns
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


class MeraLR:

  def __init__(self):
    self.m = None
    self.b = None

  def fit(self,X_train, Y_train):

    num = 0
    den = 0

    for i in range(X_train.shape[0]):
      num = num + ((X_train.iloc[i] - X_train.mean())*(Y_train.iloc[i] - Y_train.mean()))
      den = den + ((X_train.iloc[i] - X_train.mean())*(X_train.iloc[i] - X_train.mean()))

    self.m = num/den
    self.b = Y_train.mean() - (self.m * X_train.mean())
    # print(self.m)
    # print(self.b)


  def predict(self, X_test):
      return self.m * X_test + self.b
  

lr = MeraLR()

lr.fit(X_train, Y_train)

print(lr.predict(X_test.iloc[0]))

 