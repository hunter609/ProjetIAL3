from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
from datetime import datetime

def train_linear_model(X_train, y_train, start_date, end_date):
    # Check if the end date is greater than today
    today = datetime.today().strftime('%Y-%m-%d')
    if end_date > today:
        raise ValueError("La date de fin ne peut pas dépasser la date d'aujourd'hui.")
    
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
    
    # Save the training data and predictions to the data folder
    if not os.path.exists('data'):
        os.makedirs('data')
    
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    train_data.to_csv('data/training_data.csv', index=False)
    
    predictions_data = pd.DataFrame(final_predictions, columns=['predicted_prices'])
    predictions_data.to_csv('data/predictions.csv', index=False)
    
    # Prepare data to send to the front-end
    plot_data = {
        'x': list(range(len(final_predictions))),
        'y': final_predictions.flatten().tolist(),
        'type': 'scatter',
        'mode': 'lines+markers',
        'marker': {'color': 'red'},
    }
    
    plot_layout = {
        'title': 'Linear Model Predictions',
        'xaxis': {'title': 'Days'},
        'yaxis': {'title': 'Predicted Prices'},
    }
    
    # Return the model, final predictions, plot data, and plot layout for further analysis
    return model, final_predictions, plot_data, plot_layout