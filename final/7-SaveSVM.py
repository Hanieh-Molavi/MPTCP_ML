import time
import joblib
import statistics
import numpy as np
import tracemalloc
import pandas as pd
from sklearn.svm import SVC
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

folderName = r'C:/Users/Hanieh/source/final/data/4-mix-.csv'
data = pd.read_csv(folderName)

labels = data['target'].to_numpy()
features = data.drop('target', axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

model = SVC(kernel='rbf', C=0.58, gamma='auto')
model.fit(X_train, y_train)

joblib.dump(model, "4-svm_model.pkl")


avg_delay_suitable = data[data['Prediction'] == 1]['delay'].mean()  
avg_delay_unsuitable = data[data['Prediction'] == 0]['delay'].mean() 

reduction_percentage = ((avg_delay_unsuitable - avg_delay_suitable) / avg_delay_unsuitable) * 100
print(f"Percentage delay reduction: {reduction_percentage:.2f}%")


