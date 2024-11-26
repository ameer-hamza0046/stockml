import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

######################################################

class Preprocessor:
    def __init__(self, stock_df):
        self.stock_df = stock_df
    def clean_data_set(self):
        self.ndf = self.stock_df
        self.ndf.columns = self.ndf.columns.get_level_values('Price')
        return self.ndf
    def add_additional_features(self):
        df = self.ndf
        df['Day'] = df.index.day
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['Weekday'] = df.index.weekday

        def calculate_technical_indicators(df):
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close']).copy()
            
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(window=14).mean()))
            df.dropna(inplace=True)
            return df

        df = calculate_technical_indicators(df)
        df['Target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)
        df = df.dropna()
        self.ndf = df
    def plot_stock_data(self):
        self.ndf[['Close', 'Open', 'SMA_10', 'SMA_20', 'RSI']].plot(figsize=(10, 6), title='Closing Price Over Time')
        # Customize the plot
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)

        # Show the plot
        plt.savefig('Nifty-50_Data.png')
    def getTrainTestData(self, features):
        df = self.ndf
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X, y, X_train, X_test, y_train, y_test


class StockMarketPredictor:
    def __init__(self, X, y, X_train, X_test, y_train, y_test):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def logistic(self):
        model = LogisticRegression(class_weight='balanced')
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy using Logistic Regression: {accuracy:.2f}")
        return accuracy
    
    def randomForest(self, n, random_state):
        model = RandomForestClassifier(n_estimators=n, random_state=random_state, class_weight="balanced")
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy using Random Forest with {n} trees: {accuracy:.2f}")
        return accuracy

    def LSTM(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(self.X)

        def create_sequences(X, y, time_steps=10):
            X_seq, y_seq = [], []
            for i in range(len(X) - time_steps):
                X_seq.append(X[i:i+time_steps])
                y_seq.append(y[i+time_steps])
            return np.array(X_seq), np.array(y_seq)

        time_steps = 10
        X_seq, y_seq = create_sequences(X_scaled, self.y.values, time_steps)

        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(LSTM(32))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

        y_pred = (model.predict(X_test) > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy using LSTM: {accuracy:.2f}")
        return accuracy
        

RESET = '\033[0m'
GREEN = '\033[32m'
RED = '\033[31m'
if __name__ == '__main__':
    print(RED + "#### Stock Market Predictor ####")
    print(RESET + "Loading Dataset...")
    try:
        # Fetch all available data for Nifty-50
        nifty_data = yf.download("^NSEI", start="2007-01-01")
    except Exception as e:
        print(RED + "Failed")
        exit()
    print(GREEN + "Dataset Load Complete")
    print(RESET + "Preprocessing Dataset...")
    try:
        pre = Preprocessor(nifty_data)
        pre.clean_data_set()
        pre.add_additional_features()
        pre.plot_stock_data()
    except Exception as e:
        print(RED + "Failed")
        exit()
    print(GREEN + "Preprocessing Done" + RESET)
    random_state = 42
    ####################################################################
    comparison = ['Data set features', 'Only generated features', 'All the features']
    accuracies = [[], [], []]

    features = [
        ['Adj Close', 'Close', 'High', 'Low', 'Open'],
        ['SMA_10', 'SMA_20', 'RSI'],
        ['Adj Close', 'Close', 'High', 'Low', 'Open','SMA_10', 'SMA_20', 'RSI']
    ]
    i = 0
    for f in features:
        print(GREEN + "Using Features:", RESET,f)
        X, y, X_train, X_test, y_train, y_test = pre.getTrainTestData(f)
        stock_predictor = StockMarketPredictor(X, y, X_train, X_test, y_train, y_test)
        accuracies[i].append(stock_predictor.logistic())
        accuracies[i].append(stock_predictor.randomForest(1, random_state))
        accuracies[i].append(stock_predictor.randomForest(10, random_state))
        accuracies[i].append(stock_predictor.randomForest(100, random_state))
        accuracies[i].append(stock_predictor.LSTM())
        print('=======================================')
        i += 1
    
    models = ['Logi Reg',
          'RF (1 Tree)',
          'RF (10 Trees)',
          'RF (100 Trees)',
          'LSTM']
    print(GREEN + "Plotting comparison graph1: " + RESET, end='')
    x = np.arange(len(models)) 
    bar_width = 0.25 
    fig, ax = plt.subplots(figsize=(12, 6))

    # Bars for each column
    ax.bar(x - bar_width, accuracies[0], width=bar_width, label='Data Set Features', color='skyblue')
    ax.bar(x, accuracies[1], width=bar_width, label='Only Generated Features', color='orange')
    ax.bar(x + bar_width, accuracies[2], width=bar_width, label='All The Features', color='green')

    ax.set_title('Accuracy Comparison Across Experiments', fontsize=16)
    ax.set_xlabel('Models', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12,loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('result2.png',dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')

    print("Done")

    ######### another plot ################
    print(GREEN + "Plotting comparison graph2: " + RESET, end='')
    accuracy_data = accuracies

    accuracy_matrix = np.array(accuracy_data).T
    plt.figure(figsize=(12, 6))

    for i, experiment in enumerate(comparison):
        plt.plot(models, accuracy_matrix[:, i], marker='o', label=experiment)

    plt.title('Model Accuracies Across Experiments', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Experiments', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig('result1.png',dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    print("Done")