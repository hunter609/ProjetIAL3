# Description des données

Ce projet utilise des données boursières pour prédire le prix de l'or. Les données sont téléchargées à partir de Yahoo Finance et comprennent les prix de clôture quotidiens de l'or sur les 5 dernières années.

## Structure des données

- **Date**: La date à laquelle le prix a été enregistré.
- **Open**: Le prix d'ouverture de l'or pour la journée.
- **High**: Le prix le plus élevé de l'or pour la journée.
- **Low**: Le prix le plus bas de l'or pour la journée.
- **Close**: Le prix de clôture de l'or pour la journée.
- **Volume**: Le volume d'échanges de l'or pour la journée.
- **Adj Close**: Le prix de clôture ajusté pour les dividendes et les fractionnements d'actions.

## Utilisation des données

Les données sont prétraitées pour normaliser les prix et créer des ensembles de données pour l'entraînement d'un modèle LSTM. Le modèle prédit le prix de l'or dans 30 jours en utilisant les prix des 60 jours précédents comme entrée.
