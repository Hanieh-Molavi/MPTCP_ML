import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def adaptive_lasso(X, y, alpha=None):
    if alpha is None:

        lasso_cv = LassoCV(cv=5, random_state=42).fit(X, y)
        alpha = lasso_cv.alpha_

    initial_lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    initial_lasso.fit(X, y)
    
    initial_coefs = initial_lasso.coef_
    weights = 1 / (np.abs(initial_coefs) + 1e-6) 

    X_weighted = X / weights
    adaptive_lasso_model = Lasso(alpha=alpha, max_iter=10000, random_state=42)

    adaptive_lasso_model.fit(X_weighted, y)
    adaptive_coefs = adaptive_lasso_model.coef_ / weights

    return adaptive_coefs, adaptive_lasso_model


file_path = "Data/1-mix-.csv"  
data = pd.read_csv(file_path)

target_column = "target"  
y = data[target_column].values

X = data.drop(columns=[target_column]).values
scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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