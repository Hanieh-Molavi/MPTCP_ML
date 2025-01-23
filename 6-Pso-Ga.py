import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO
from geneticalgorithm import geneticalgorithm as GA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd


def ga_fitness(params):
    return svm_fitness(params)

def svm_fitness(params):
    C, gamma = params
    if C <= 0 or gamma <= 0:
        return float('inf') 
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, X_train_reduced, y_train, cv=5, scoring='accuracy')
    return -scores.mean() 

file_path = "3-mix-.csv"  
df = pd.read_csv(file_path)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train_scaled)

X_test_reduced = pca.transform(X_test_scaled)
print("Running PSO...")
pso = PSO(func=svm_fitness, dim=2, pop=30, max_iter=20, lb=[0.1, 0.0001], ub=[100, 1], w=0.5, c1=1.5, c2=1.5)

pso.run()
pso_best_params = pso.gbest_x

pso_best_score = -pso.gbest_y
pso_best_y_hist = -np.array(pso.gbest_y_hist)
print("PSO Best Parameters:", pso_best_params)

print("PSO Best Score:", pso_best_score)
print("Running GA...")

ga_model = GA(
    function=ga_fitness,
    dimension=2,
    variable_type='real',
    variable_boundaries=np.array([[0.01, 100], [0.0001, 1]]),
    algorithm_parameters={
        'max_num_iteration': 50,
        'population_size': 20,
        'mutation_probability': 0.2,
        'elit_ratio': 0.01,
        'parents_portion': 0.3,
        'crossover_probability': 0.8,
        'max_iteration_without_improv': None
    }
)


initial_population = np.random.uniform(
    low=[pso_best_params[0] * 0.8, pso_best_params[1] * 0.8],
    high=[pso_best_params[0] * 1.2, pso_best_params[1] * 1.2],
    size=(20, 2)
)

ga_model.population = initial_population
ga_model.run()

ga_best_params = ga_model.output_dict['variable']
ga_best_score = -ga_model.output_dict['function']
ga_best_y_hist = -np.array(ga_model.report) 

print("GA Best Parameters:", ga_best_params)
print("GA Best Score:", ga_best_score)

final_model = SVC(C=ga_best_params[0], gamma=ga_best_params[1], kernel='rbf')
final_model.fit(X_train_reduced, y_train)

final_accuracy = final_model.score(X_test_reduced, y_test)
print("Final Model Accuracy on Test Set:", final_accuracy)


plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pso_best_y_hist) + 1), pso_best_y_hist, label="PSO Progress", marker='o', linestyle='-')
plt.plot(range(1, len(ga_best_y_hist) + 1), ga_best_y_hist, label="GA Progress", marker='s', linestyle='--')

plt.title("Optimization Progress: PSO vs GA", fontsize=16)
plt.xlabel("Iteration (Generation)", fontsize=14)
plt.ylabel("Best Objective Function Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()