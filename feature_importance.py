import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import seaborn as sns

# Define a wrapper class for Keras models to be compatible with scikit-learn
class KerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn):
        self.build_fn = build_fn
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn(input_dim=X.shape[1])
        self.model.fit(X, y, epochs=50, batch_size=16, verbose=0)  # Moderately increased epochs
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

# Function to build a moderately simple neural network
def build_nn(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(8, activation='relu'))  # First hidden layer with 32 neurons
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(8, activation='relu'))  # Second hidden layer with 16 neurons
    model.add(Dense(1, activation='linear'))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }

# Load the input features
data = pd.read_csv('data_ml/input_features/combined_input_features_3.csv')

# Define the target variable (y) and the features (X)
target = 'Tour Length'
features = data.columns.difference([target])

# Separate the features and target variable
X = data[features]
y = data[target]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# Define models
rf = RandomForestRegressor(n_estimators=50, random_state=42)  # Slightly increased trees
svr = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2', n_jobs=-1)
nn = KerasRegressor(build_fn=build_nn)  # Improved Neural Network

# Train Random Forest
rf.fit(X_train_scaled, y_train)

# Train SVR
svr.fit(X_train_scaled, y_train)

# Train Neural Network using the wrapper
nn.fit(X_train_scaled, y_train)

# Evaluate models
models = {
    'RandomForest': rf,
    'SVR': svr,
    'NeuralNetwork': nn
}


def plot_feature_importances(importances, std, indices, model_name, features):
    plt.figure(figsize=(14, 8))
    plt.title(f"Feature Importances: {model_name}", fontsize=16, fontweight='bold')

    # Use a color gradient for bars
    colors = sns.color_palette("viridis", len(importances))

    # Plot with color gradient
    plt.bar(range(len(importances)), importances[indices], color=colors, yerr=std[indices], align="center")

    # Add value labels on the bars
    for i, (imp, std) in enumerate(zip(importances[indices], std[indices])):
        plt.text(i, imp + std, f'{imp:.3f}', ha='center', va='bottom', fontsize=9)

    # Improve x-ticks with better rotation and font size
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right', fontsize=10)
    plt.xlim([-1, len(importances)])

    # Set labels and grid
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# Evaluate and plot feature importances
for model_name, model in models.items():
    y_valid_pred = model.predict(X_valid_scaled)

    mse = mean_squared_error(y_valid, y_valid_pred)
    r2 = r2_score(y_valid, y_valid_pred)
    print(f'{model_name} - MSE: {mse}, R2: {r2}')

    # Calculate Permutation Importance
    results = permutation_importance(model, X_valid_scaled, y_valid, n_repeats=10, random_state=42, n_jobs=-1)

    importances = results.importances_mean
    std = results.importances_std
    indices = np.argsort(importances)[::-1]

    # Print feature importances
    print(f"\nFeature importances for {model_name}:")
    for i in indices:
        print(f"Feature: {features[i]}, Importance: {importances[i]:.3f}")

    # Plot the enhanced feature importances
    plot_feature_importances(importances, std, indices, model_name, features)
