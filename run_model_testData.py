import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random_forest import RandomForestModel
from neural_network import NeuralNetworkModel
from svr import SupportVectorRegressorModel
from gbm import GradientBoostingModel
from feature_manager import FeatureManager
import numpy as np
import os
from sklearn.impute import SimpleImputer


def main():
    # Load data
    # Load data
    train_data = pd.read_csv('data_ml/input_features/combined_input_features_3.csv')
    test_data = pd.read_csv('data_ml/input_features/combined_input_features_1.csv')

    # Initialize feature manager and models
    feature_manager = FeatureManager()
    models = {
        'RandomForest': RandomForestModel(),
        'NeuralNetwork': NeuralNetworkModel(),
        'SVR': SupportVectorRegressorModel(),
        'GradientBoosting': GradientBoostingModel()
    }
    results = []
    for feature_set_name, feature_set in feature_manager.feature_sets.items():
        # Define the target variable (y) and the features (X)
        target = 'Tour Length'
        all_features = train_data.columns.difference([target])

        # Replace -1 with NaN and handle missing values
        train_data.replace([-1], np.nan, inplace=True)
        test_data.replace([-1], np.nan, inplace=True)


        features = list(set(feature_set).intersection(set(all_features)))

        X_train = train_data[features]
        y_train = train_data[target]
        X_test = test_data[features]
        y_test = test_data[target]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Split the data into training and validation sets
       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in models.items():
            model.train(X_train_scaled, y_train)
            mse, r2, y_pred = model.evaluate(X_test_scaled, y_test)
            results.append({
                'Feature Set': feature_set_name,
                'Model': model_name,
                'MSE': mse,
                'R2': r2,
            })
            # Save predictions
            # save_predictions(feature_set_name, model_name, y_valid, y_pred)
    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('data_ml/model_results/model_evaluation_results_test_set.csv', index=False)
    print("Results saved")


if __name__ == "__main__":
    main()
