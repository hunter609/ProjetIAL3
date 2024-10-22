from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from models.linear_model import train_linear_model
from datetime import datetime, timedelta
import io
import base64
import threading
import sqlite3

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Initialize SQLite database for view count
def init_db():
    conn = sqlite3.connect('view_count.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS view_count (count INTEGER)''')
    c.execute('''INSERT INTO view_count (count) SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM view_count)''')
    conn.commit()
    conn.close()

init_db()

# Télécharger les données boursières d'or
def load_data(start_date, end_date):
    df = yf.download('GC=F', start=start_date, end=end_date)
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

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    prediction_step = int(content.get('prediction_step', 30))  # Assurer que c'est un entier
    start_date = content.get('start_date', '2019-01-01')
    end_date = content.get('end_date', datetime.today().strftime('%Y-%m-%d'))

    # Vérifier si la date de fin dépasse aujourd'hui
    if datetime.strptime(end_date, '%Y-%m-%d') > datetime.today():
        return jsonify({'error': 'La date de fin ne peut pas dépasser la date d\'aujourd\'hui.'}), 400
    
    df = load_data(start_date, end_date)
    scaled_data, scaler = preprocess_data(df)

    # Créer les ensembles de données
    time_step = 60
    X, y = create_dataset(scaled_data, time_step, prediction_step)
    X_linear = X.reshape(X.shape[0], X.shape[1])  # Reshape pour le modèle linéaire

    # Entraîner le modèle linéaire
    model_linear, final_predictions, plot_data, plot_layout = train_linear_model(X_linear, y, start_date, end_date)  # Ajuster le déballage

    # Prédire le prix pour les prochains jours
    last_60_days = scaled_data[-time_step:]  # Derniers 60 jours
    last_60_days_linear = last_60_days.reshape(1, time_step)  # Préparer l'entrée pour le modèle linéaire
    
    predictions_linear = []
    predicted_prices = []  # Liste pour stocker les prix prédits
    
    for day in range(prediction_step):  # Prédire pour chaque jour de 0 à prediction_step-1
        prediction_linear = model_linear.predict(last_60_days_linear)
        predictions_linear.append(prediction_linear[0])  # Stocker la valeur de prédiction linéaire directement
        predicted_prices.append(scaler.inverse_transform(prediction_linear.reshape(-1, 1))[0][0])  # Stocker la prédiction de prix réelle
        last_60_days_linear = np.append(last_60_days_linear[:, 1:], prediction_linear.reshape(1, 1), axis=1)

    predictions_linear = scaler.inverse_transform(np.array(predictions_linear).reshape(-1, 1))

    # Visualisation
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close'], color='blue', label='Prix réel', linewidth=2)
    plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Date de prévision', linewidth=2)

    # Continuer la ligne bleue pour les jours de prévision
    future_dates = []
    current_date = df.index[-1]
    while len(future_dates) < prediction_step:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Exclude Saturdays and Sundays
            future_dates.append(current_date)
    
    plt.plot(future_dates, predictions_linear, color='purple', label=f'Prévisions Linéaires dans {prediction_step} jours', linewidth=2)

    # Ajouter le point orange
    scatter = plt.scatter(future_dates, predictions_linear, color='orange', label='Prévisions', s=20)

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
        text = f'{predicted_prices[ind["ind"][0]]:.2f}\n{future_dates[ind["ind"][0]].strftime("%Y-%m-%d")}'
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
    plt.title('Prévision du Prix de l\'Or', fontsize=16)
    plt.xlabel('Temps', fontsize=14)
    plt.ylabel('Prix des Actions (USD)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

     # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode()
    # Convert plot to data for front-end
    plot_data = {
        'plot_data': [
            {
                'x': df.index.to_list() + [date.strftime('%Y-%m-%d') for date in future_dates],
                'y': df['Close'].to_list() + predictions_linear.flatten().tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Prix réel et prévisions'
            }
        ],
        'plot_layout': {
            'title': 'Prévision du Prix de l\'Or',
            'xaxis': {'title': 'Temps'},
            'yaxis': {'title': 'Prix des Actions (USD)'},
            'shapes': [
                {
                    'type': 'line',
                    'x0': df.index[-1],
                    'y0': min(df['Close']),
                    'x1': df.index[-1],
                    'y1': max(df['Close']),
                    'line': {
                        'color': 'red',
                        'width': 2,
                        'dash': 'dashdot',
                    },
                }
            ]
        }
    }

    # Increment view count
    conn = sqlite3.connect('view_count.db')
    c = conn.cursor()
    c.execute('''UPDATE view_count SET count = count + 1''')
    conn.commit()
    c.execute('''SELECT count FROM view_count''')
    view_count = c.fetchone()[0]
    conn.close()

    return jsonify({'plot_url': plot_url, 'plot_data': plot_data['plot_data'], 'plot_layout': plot_data['plot_layout'], 'view_count': view_count})

@app.route('/view-count', methods=['GET'])
def get_view_count():
    conn = sqlite3.connect('view_count.db')
    c = conn.cursor()
    c.execute('''SELECT count FROM view_count''')
    view_count = c.fetchone()[0]
    conn.close()
    return jsonify({'view_count': view_count})

if __name__ == "__main__":
    app.run(debug=True)
