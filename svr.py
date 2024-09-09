from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

class SupportVectorRegressorModel:
    def __init__(self):
        # Initialize SVR with default parameters
        self.model = SVR(kernel='rbf')

    def train(self, X_train, y_train):
        # Define parameter grid for SVR
        param_grid = {
            'C': [0.1, 1, 10],  # Regularization parameter
            'epsilon': [0.01, 0.1, 0.2],  # Epsilon in the loss function
            'gamma': ['scale', 0.1, 0.01]  # Kernel coefficient
        }

        # Use GridSearchCV to find the best parameters
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Set the best estimator as the model
        self.model = grid_search.best_estimator_

        # Print best parameters found by grid search
        print(f"Best parameters found: {grid_search.best_params_}")

    def evaluate(self, X_valid, y_valid, save_path=None):
        # Make predictions on the validation set
        y_pred = self.model.predict(X_valid).flatten()
        # Calculate MSE and R²
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)

        # Save results if a save path is provided
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
        # Generate a scatter plot of actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_valid, y_pred, alpha=0.3, color='blue', label='Predicted')
        plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2, label='Ideal')
        plt.xlabel('Actual Tour Length')
        plt.ylabel('Predicted Tour Length')
        plt.title('Actual vs Predicted Tour Length')
        plt.legend()
        plt.grid(True)

        # Save the plot if a save path is provided
        if save_path:
            plot_save_path = save_path.replace('.csv', '.png')
            plt.savefig(plot_save_path)
            print(f"Plot saved to {plot_save_path}")
        else:
            plt.show()

# Example usage
# svr_model = SimpleSVRModel()
# svr_model.train(X_train, y_train)
# mse, r2, y_pred = svr_model.evaluate(X_valid, y_valid)
