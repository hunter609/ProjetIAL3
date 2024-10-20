import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from models.linear_model import train_linear_model  # Importing the linear model
from datetime import datetime

# Télécharger les données boursières d'or (tsy maintsy start='2024-01-01' amin'izay precis tsara)
def load_data():
    today = datetime.today().strftime('%Y-%m-%d')
    print(today)
    df = yf.download('GC=F', start='2019-01-01', end='2024-10-20')
    # df = yf.download('GC=F', start='2024-01-01', end=today)
    return df

# Prétraiter les données
def preprocess_data(df):
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Créer des ensembles de données pour le modèle
def create_dataset(data, time_step=60, prediction_step=30):
    X, y = [], []
    for i in range(len(data) - time_step - prediction_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step + prediction_step - 1, 0]) 
    return np.array(X), np.array(y)

def main():
    df = load_data()
    scaled_data, scaler = preprocess_data(df)

    # Créer les ensembles de données
    time_step = 60
    prediction_step = 30  # Prédire le prix dans 30 jours (environ 1 mois)
    X, y = create_dataset(scaled_data, time_step, prediction_step)
    X_linear = X.reshape(X.shape[0], X.shape[1])  # Reshape for linear model

    # Entraîner le modèle linéaire
    model_linear, _ = train_linear_model(X_linear, y)  # Train linear model

    # Prédire le prix pour les 30 prochains jours
    last_60_days = scaled_data[-time_step:]  # Derniers 60 jours
    last_60_days_linear = last_60_days.reshape(1, time_step)  # Prepare input for linear model
    
    predictions_linear = []
    predicted_prices = []  # List to store predicted prices
    
    for day in range(30):  # Prédire pour chaque jour de 0 à 29
        prediction_linear = model_linear.predict(last_60_days_linear)
        predictions_linear.append(prediction_linear[0])  # Store the linear prediction value directly
        predicted_prices.append(scaler.inverse_transform(prediction_linear.reshape(-1, 1))[0][0])  # Store the actual price prediction
        last_60_days_linear = np.append(last_60_days_linear[:, 1:], prediction_linear.reshape(1, 1), axis=1)

    predictions_linear = scaler.inverse_transform(np.array(predictions_linear).reshape(-1, 1))

    # Visualisation
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close'], color='blue', label='Prix réel', linewidth=2)
    plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Date de prévision', linewidth=2)

    # Continuer la ligne bleue pour les 30 jours de prévision
    future_dates = [df.index[-1] + pd.Timedelta(days=i + 1) for i in range(30)]
    plt.plot(future_dates, predictions_linear, color='purple', label='Prévisions Linéaires dans 30 jours', linewidth=2)  # Nouvelle couleur pour la ligne de prévision linéaire

    # Ajouter le point orange
    scatter = plt.scatter(future_dates, predictions_linear, color='orange', label='Prévisions', s=20)  # Point orange pour les prévisions

    # Afficher les prix prévus avec un peu d'espace pour éviter la superposition
    for i, price in enumerate(predicted_prices):
        if i == len(predicted_prices) - 1:
            plt.annotate(f'{price:.2f}', (future_dates[i], predictions_linear[i]), textcoords="offset points", xytext=(0,15), ha='center', fontsize=8, color='orange')

    # Fonction pour afficher le prix prédit au survol
    annot = plt.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="w"),
                         arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f'{predicted_prices[ind["ind"][0]]:.2f}'
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == plt.gca():
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                plt.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    plt.draw()

    plt.gcf().canvas.mpl_connect("motion_notify_event", hover)

    # Améliorer les axes
    plt.title('Prévision du Prix de l\'Or', fontsize=16)  # Changed title to reflect gold price prediction
    plt.xlabel('Temps', fontsize=14)
    plt.ylabel('Prix des Actions (USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
