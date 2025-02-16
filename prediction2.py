# libraries
import streamlit as st
import pickle
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.optimize import minimize
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Game theory approach for preprocessing
def game_theory_feature_selection(X, y, models):
    n_features = X.shape[1]
    n_models = len(models)

    def payoff_function(weights):
        weights = np.array(weights)
        combined_score = 0
        for i, model in enumerate(models):
            model.fit(X * weights, y)
            score = model.score(X * weights, y)
            combined_score += score / n_models
        return -combined_score  # We minimize, so we take the negative

    # Initial weights for features
    initial_weights = np.ones(n_features) / n_features
    bounds = [(0, 1) for _ in range(n_features)]
    result = minimize(payoff_function, initial_weights, bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    return result.x

def prediction2():
    st.title('Prediction Of Covid-19 with Game Theory Optimization')
    # Load dataset
    dataset = pd.read_csv("covid_19_state_wise_data.csv")

    # Preprocessing
    label_encoder = LabelEncoder()
    if "state" in dataset.columns:
        dataset["state"] = label_encoder.fit_transform(dataset["state"])

    # Split dataset into features and target
    features = ["confirmed_cases", "active_cases", "recovered_cases", "death_cases"]
    X = dataset[features]

    targets = ["survival_rate"]

    # Train the models
    linear_regression = LinearRegression()
    random_forest = RandomForestRegressor()

    models = [linear_regression, random_forest]

    for target in targets:
        if target in dataset.columns:
            y = dataset[target]
            
            # Apply game theory for feature weighting
            optimized_weights = game_theory_feature_selection(X, y, models)
            st.write(f"Optimized Feature Weights: {optimized_weights}")
            
            # Train models using optimized feature set
            X_optimized = X * optimized_weights
            
            # Fit models with optimized features
            linear_regression.fit(X_optimized, y)
            random_forest.fit(X_optimized, y)

            # Predict
            predictions_linear_regression = linear_regression.predict(X_optimized)
            predictions_random_forest = random_forest.predict(X_optimized)

            # Display the predictions
            st.write(f"Predictions for {target} after applying Game Theory:")
            st.write("Linear Regression:", predictions_linear_regression)
            st.write("Random Forest:", predictions_random_forest)

    # Specify the path to your dataset.csv file
    data_path = 'WHO-COVID-19-global-data.csv'

    # Read the dataset.csv file
    df = pd.read_csv(data_path)

    # Convert the 'Date_reported' column to datetime
    df['Date_reported'] = pd.to_datetime(df['Date_reported'])

    # Filter the columns of interest
    df = df[['Date_reported', 'New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']]

    # Set the 'Date_reported' column as the index
    df.set_index('Date_reported', inplace=True)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Create sequences
    def create_sequences(data, seq_length):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    sequence_length = 10
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build and compile LSTM model
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, X_train.shape[2])),
        Dense(32, activation='relu'),
        Dense(X_train.shape[2])
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    st.write('Test Loss:', loss)
    accuracy = 100 - loss * 100
    st.write('Accuracy after applying Game Theory:', accuracy)

    # Predict for the next 1 year
    last_sequence = test_data[-sequence_length:]
    predictions = []
    for _ in range(365):
        predicted = model.predict(np.expand_dims(last_sequence, axis=0))
        predictions.append(predicted)
        last_sequence = np.concatenate((last_sequence[1:], predicted), axis=0)

    predictions = np.array(predictions).reshape(-1, X_train.shape[2])
    predictions = scaler.inverse_transform(predictions)

    dates = pd.date_range(start=df.index[-1], periods=len(predictions))
    predicted_df = pd.DataFrame(predictions, index=dates, columns=df.columns)

    st.write(predicted_df)

    plt.figure(figsize=(12, 6))
    for column in predicted_df.columns:
        plt.plot(predicted_df.index, predicted_df[column], linestyle='dotted', label=column)

    plt.title('Predicted COVID-19 Data for the Next 1 Year with Game Theory')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    st.pyplot(plt)
