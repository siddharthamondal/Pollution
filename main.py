import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load data
data = pd.read_excel('new_data.xlsx')

# Clean data
data = data.replace('None', np.nan)
data = data.dropna()

# Create separate models for each station
stations = data['Station'].unique()

for station in stations:
    # Filter data for the current station
    station_data = data[data['Station'] == station]

    # Select the relevant features
    X = station_data[['PM10', 'NO', 'NO2', 'NOx', 'SO2', 'CO', 'Ozone', 'WS', 'WD', 'SR', 'AT']]
    y = station_data['PM2.5']

    # Scale the data
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

    # Reshape the data for the LSTM model
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0, shuffle=False)

    # Make predictions for the next season
    next_season_data = station_data.tail(11)[['PM10', 'NO', 'NO2', 'NOx', 'SO2', 'CO', 'Ozone', 'WS', 'WD', 'SR', 'AT']]
    next_season_data_scaled = scaler_x.transform(next_season_data)
    next_season_data_scaled = next_season_data_scaled.reshape(
        (next_season_data_scaled.shape[0], 1, next_season_data_scaled.shape[1]))
    next_season_predictions_scaled = model.predict(next_season_data_scaled)
    next_season_predictions = scaler_y.inverse_transform(next_season_predictions_scaled)

    # Print the predictions for the next season
    print(station)
    print(next_season_predictions.flatten()[0])