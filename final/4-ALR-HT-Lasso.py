import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def adaptive_lasso(X, y, alpha=None):
    """
    Perform Adaptive LASSO Regression.

    Parameters:
    - X: np.ndarray
        Feature matrix (n_samples, n_features).
    - y: np.ndarray
        Response vector (n_samples,).
    - alpha: float or None
        Regularization parameter. If None, uses LassoCV for automatic selection.

    Returns:
    - coef: np.ndarray
        Coefficients of the Adaptive LASSO model.
    - model: Lasso
        Trained LASSO model.
    """
    # Step 1: Fit an initial LASSO to get preliminary coefficients
    if alpha is None:
        lasso_cv = LassoCV(cv=5, random_state=42).fit(X, y)
        alpha = lasso_cv.alpha_

    initial_lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    initial_lasso.fit(X, y)
    initial_coefs = initial_lasso.coef_

    # Step 2: Calculate weights for Adaptive LASSO
    weights = 1 / (np.abs(initial_coefs) + 1e-6)  # Avoid division by zero

    # Step 3: Apply weights and refit LASSO
    X_weighted = X / weights
    adaptive_lasso_model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    adaptive_lasso_model.fit(X_weighted, y)

    # Adjust coefficients back
    adaptive_coefs = adaptive_lasso_model.coef_ / weights

    return adaptive_coefs, adaptive_lasso_model


file_path = "Data/1-mix-.csv"  
data = pd.read_csv(file_path)

# Define target (y) and features (X)
target_column = "target"  # Replace with your target column name
y = data[target_column].values
X = data.drop(columns=[target_column]).values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run Adaptive LASSO
coef, model = adaptive_lasso(X_train, y_train)

print("Adaptive LASSO Coefficients:")
print(coef)

model.fit(X_train, y_train)

selected_features = X.columns[(model.coef_ != 0)]
features= []

for i in selected_features:
    features.append(str(i))

features.append('target')
selected_columns = data[selected_features]

selected_columns['target'] = y
selected_columns.to_csv('1-mix-.csv', index=False)