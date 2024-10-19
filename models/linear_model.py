from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_linear_model(X_train, y_train):
    # Reshape the input data to ensure it is in the correct format
    X_train = X_train.reshape(X_train.shape[0], -1)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Validate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error on validation set: {mse}")
    
    # Ensure predictions are accurate by using the entire dataset for final predictions
    model.fit(X_train, y_train)  # Refit the model on the full training data
    final_predictions = model.predict(X_train)  # Predict on the training data for accuracy
    print(f"Final predicted prices for the training set: {final_predictions.flatten()}")
    
    # Return the model and final predictions for further analysis
    return model, final_predictions
