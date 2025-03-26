from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def evaluate_predictions(df: pd.DataFrame):
    """
    Evaluate the predictions using RMSE and R-squared metrics.
    Args:
        df (pd.DataFrame): DataFrame containing the ground truth and predicted values.
    """
    # Compute RMSE
    rmse = root_mean_squared_error(df["y"], df["y_hat"])
    print(f"RMSE: {rmse:.4f}")

    # Compute R-squared
    r2 = r2_score(df["y"], df["y_hat"])
    print(f"R-squared: {r2:.4f}")

    # Convert to NumPy for calculations
    y = np.array(df["y"])
    y_hat = np.array(df["y_hat"])

    # Scatter plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=y, y=y_hat)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal Fit (Y=X)")

    # Annotate R^2 value
    plt.text(0.05, 0.92, f"$R^2 = {r2:.3f}$", fontsize=15, color="black")

    # Labels
    plt.xlabel("Ground Truth Severity", fontsize=12)
    plt.ylabel("Predicted Severity", fontsize=12)
    plt.xlim(right=1.02, left=-0.02)
    plt.ylim(top=1.02, bottom=-0.02)
    plt.grid()

    plt.show()