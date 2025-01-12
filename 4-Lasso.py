import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

for j in range(1,5):
    name = str(j) + "-mix-.csv"
    
    data = pd.read_csv("C:/Users/Hanieh/source/final/"+ name)
    X = data.drop(columns=['target'])  

    y = data['target']
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lasso = Lasso(alpha=0.01, max_iter=1000) 
    lasso.fit(X_train, y_train)

    selected_features = X.columns[(lasso.coef_ != 0)]
    features= []

    for i in selected_features:
        features.append(str(i))

    features.append('target')
    selected_columns = data[selected_features]

    selected_columns['target'] = y
    selected_columns.to_csv(name, index=False)