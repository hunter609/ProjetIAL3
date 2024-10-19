from sklearn.linear_model import LinearRegression
import numpy as np

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return model
