from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def create_and_train_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))  # Increased dropout for better regularization
    model.add(LSTM(units=150, return_sequences=True))
    model.add(Dropout(0.3))  # Increased dropout for better regularization
    model.add(LSTM(units=150))
    model.add(Dropout(0.3))  # Increased dropout for better regularization
    model.add(Dense(units=100))  # Increased units for more complexity
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Implementing early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=[early_stopping])  # Increased epochs for better training
    
    return model
