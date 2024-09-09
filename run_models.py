import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random_forest import RandomForestModel
#from neural_network import NeuralNetworkModel
#from svr import SupportVectorRegressorModel
#from rf_hyper import RandomForestModel
from nn_hyper import NeuralNetworkModel
from svr_hyper import SupportVectorRegressorModel

from gbm import GradientBoostingModel
from feature_manager import FeatureManager
import numpy as np
import os
from sklearn.impute import SimpleImputer

def main():
    # Load data
    # Load data
    data = pd.read_csv('data_ml/input_features/combined_input_features_521.csv')

    # Initialize feature manager and models
    feature_manager = FeatureManager()
    models = {
        'RandomForest': RandomForestModel(),
        'NeuralNetwork': NeuralNetworkModel(),
        'SVR': SupportVectorRegressorModel()
       #'GradientBoosting': GradientBoostingModel()
    }
    results = []

    prediction_dir = 'data_ml/prediction_visualization/Test521-4'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    for feature_set_name, feature_set in feature_manager.feature_sets.items():
        # Define the target variable (y) and the features (X)
        target = 'Tour Length'
        all_features = data.columns.difference([target])

        # Replace -1 with NaN and handle missing values
        #data.replace([-1], np.nan, inplace=True)
        features = list(set(feature_set).intersection(set(all_features)))
        X = data[features]
        y = data[target]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        # Split the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        for model_name, model in models.items():
            model.train(X_train_scaled, y_train)
            save_path = os.path.join(prediction_dir, f'{model_name}_{feature_set_name}_predictions.csv')
            mse, r2, y_pred = model.evaluate(X_valid_scaled, y_valid, save_path=save_path)
            results.append({
                'Feature Set': feature_set_name,
                'Model': model_name,
                'MSE': mse,
                'R2': r2,
            })
    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('data_ml/model_results/model_evaluation_results_521-4.csv', index=False)
    print("Results saved")

if __name__ == "__main__":
    main()
