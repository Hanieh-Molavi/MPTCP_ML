import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sko.PSO import PSO
from geneticalgorithm import geneticalgorithm as GA
import numpy as np

# Load dataset
file_path = "3-mix-.csv"
df = pd.read_csv(file_path)
print("read data")

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("train")
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train_scaled)
X_test_reduced = pca.transform(X_test_scaled)
print("pca")

def svm_fitness_reduced(params):
    print("svm_fitness_reduced")
    C, gamma = params
    if C <= 0 or gamma <= 0:
        print("==> C:",C,"\n==> Gamma:",gamma)
        return float('inf')
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, X_train_reduced, y_train, cv=5, scoring='accuracy')
    print("==> scores:",scores)
    return -scores.mean()

pso = PSO(func=svm_fitness_reduced, dim=2, pop=10000, max_iter=12, lb=[0.1, 0.0001], ub=[100, 1], w=0.5, c1=1.6, c2=1.4)
pso.run()
print("pso.run")
print("==> pso:",pso)
# PSO Results
pso_best_params = pso.gbest_x
pso_best_score = -pso.gbest_y
print("Best Score per Iteration (PSO):", pso.gbest_y_hist)
# GA Optimization
def ga_fitness_reduced(params):
    print("ga_fitness_reduced")
    print("==> ga params:",params)
    return svm_fitness_reduced(params)

ga_model = GA(
    function=ga_fitness_reduced,
    dimension=2,
    variable_type='real',
    variable_boundaries=np.array([[0.01, 100], [0.0001, 1]],
    mutation_probability=0.5)
)
ga_model.run()
print("ga run")
print("==> ga:",ga_model)
print("Best Score per Generation (GA):", ga_model.report)
# GA Results
ga_best_params = ga_model.output_dict['variable']
ga_best_score = -ga_model.output_dict['function']

# Train final SVM model with best parameters
final_model = SVC(C=ga_best_params[0], gamma=ga_best_params[1], kernel='rbf')
final_model.fit(X_train_reduced, y_train)
final_accuracy = final_model.score(X_test_reduced, y_test)
print("final")
# Results
print("Best Parameters from PSO:", pso_best_params)
print("Best Score from PSO:", pso_best_score)
print("Best Parameters from GA:", ga_best_params)
print("Best Score from GA:", ga_best_score)
print("Final Model Accuracy on Test Set with Reduced Data:", final_accuracy)