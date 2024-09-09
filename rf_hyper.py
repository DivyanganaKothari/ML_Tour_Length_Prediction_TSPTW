from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.feature_set_name = None

    def train(self, X_train, y_train):
        # Define parameter grid for Random Forest
        param_grid = {
            'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': ['sqrt', 'log2']  # Number of features to consider when looking for the best split
        }

        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Set the best estimator as the model
        self.model = grid_search.best_estimator_

        # Print best parameters found by grid search
        print(f"Best parameters found: {grid_search.best_params_}")

    def evaluate(self, X_valid, y_valid, save_path=None):
        y_pred = self.model.predict(X_valid).flatten()
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)

        if save_path:
            results_df = pd.DataFrame({
                'y_valid': y_valid,
                'y_pred': y_pred,
            })
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        # Plotting the results
        self.plot_predictions(y_valid, y_pred, save_path)

        return mse, r2, y_pred

    def plot_predictions(self, y_valid, y_pred, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_valid, y_pred, alpha=0.3, color='blue', label='Predicted')
        plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2, label='Ideal')
        plt.xlabel('Actual Tour Length')
        plt.ylabel('Predicted Tour Length')
        plt.title('Actual vs Predicted Tour Length')
        plt.legend()
        plt.grid(True)

        # Save the plot if save_path is provided
        if save_path:
            plot_save_path = save_path.replace('.csv', '.png')
            plt.savefig(plot_save_path)
            print(f"Plot saved to {plot_save_path}")
        else:
            plt.show()