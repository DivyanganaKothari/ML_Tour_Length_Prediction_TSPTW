from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

class SupportVectorRegressorModel:
    def __init__(self):
        self.model = None
        self.best_params = None
    def train(self, X_train, y_train):
        # Perform grid search to tune hyperparameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }

        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best hyperparameters
        self.best_params = grid_search.best_params_
        print("Best hyperparameters:", self.best_params)

        # Train the best SVR model
        self.model = grid_search.best_estimator_
        self.model.fit(X_train, y_train)

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