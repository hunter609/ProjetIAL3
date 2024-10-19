import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from models.lstm_model import create_and_train_lstm

def load_data():
    df = yf.download('GC=F', start='2024-01-01', end='2024-09-30')
    return df

def preprocess_data(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def predict_future_prices(model, scaled_data, scaler, days=30):
    predictions = []
    last_60_days = scaled_data[-60:]  # Derniers 60 jours
    last_60_days = last_60_days.reshape((1, 60, 1))

    for _ in range(days):
        prediction = model.predict(last_60_days)
        predictions.append(prediction[0][0])
        
        # Mettre à jour last_60_days avec la nouvelle prédiction
        last_60_days = np.concatenate((last_60_days[:, 1:, :], prediction.reshape(1, 1, 1)), axis=1)

    # Inverser la transformation des prévisions
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def main():
    df = load_data()
    scaled_data, scaler = preprocess_data(df)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model_lstm = create_and_train_lstm(X, y)

    # Prédire les prix pour le mois suivant
    future_prices = predict_future_prices(model_lstm, scaled_data, scaler, days=30)

    # Visualiser les résultats
    plt.figure(figsize=(14, 5))
    plt.plot(df['Close'], color='blue', label='Prix réel')
    
    # Ajouter les prévisions futures
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    plt.plot(future_dates, future_prices, color='green', label='Prévisions pour le mois suivant')
    
    plt.title('Prévisions des Cours des Actions')
    plt.xlabel('Temps')
    plt.ylabel('Prix des Actions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
