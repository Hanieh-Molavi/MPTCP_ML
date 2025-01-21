import time
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

def SVM(file,fileCounter):
    data = pd.read_csv(file)

    labels = data['target'].to_numpy()
    features = data.drop('target', axis=1).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='rbf', C=0.51, gamma='auto')
    svm_classifier.fit(X_train, y_train)

    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    y_train_pred = svm_classifier.predict(X_train)
    y_test_pred = svm_classifier.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Accuracy on Training Data:", train_accuracy)
    print("Accuracy on Test Data:", test_accuracy)

    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)

    TP = np.diag(conf_matrix) 
    FP = conf_matrix.sum(axis=0) - TP 
    
    FN = conf_matrix.sum(axis=1) - TP 
    TN = conf_matrix.sum() - (FP + FN + TP)

    pca = PCA(n_components=3)

    X_reduced = pca.fit_transform(features)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels, cmap='viridis', alpha=0.5)

    predictions_reduced = pca.transform(X_test)
    ax.scatter(predictions_reduced[:, 0], predictions_reduced[:, 1], predictions_reduced[:, 2], 
            c='red', marker='x', label='Predictions')


    if train_accuracy > test_accuracy and (train_accuracy - test_accuracy) > 0.1:
        print("The model might be overfitting.")
    elif train_accuracy < 0.8 and test_accuracy < 0.8:
        print("The model might be underfitting.")
    else:
        print("The model seems to have a good balance between train and test accuracy.")
    fig.colorbar(scatter, ax=ax, label='Cluster Label')
    ax.legend()

    ax.set_title('3D Visualization of Clusters and Predictions')
    ax.set_xlabel('PCA Component 1')

    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')

    plt.show()
    plt.savefig(str(fileCounter) + '_svm')

    # labels_classes = [f'Class {i}' for i in range(len(TP))]  
    # bar_width = 0.2

    #x = np.arange(len(labels_classes))
    #fig, ax = plt.subplots()

    # bar1 = ax.bar(x - bar_width * 1.5, TP, bar_width, label='True Positives (TP)')
    # bar2 = ax.bar(x - bar_width / 2, FP, bar_width, label='False Positives (FP)')

    # bar3 = ax.bar(x + bar_width / 2, FN, bar_width, label='False Negatives (FN)')
    # bar4 = ax.bar(x + bar_width * 1.5, TN, bar_width, label='True Negatives (TN)')

    # ax.set_xlabel('Classes')
    # ax.set_ylabel('Count')

    # ax.set_title('TP, FP, FN, TN for Each Class')
    # ax.set_xticks(x)

    # ax.set_xticklabels(labels_classes)
    # ax.legend()

    # plt.savefig(str(fileCounter) + '_fscore')
    return accuracy,f1,TP,TN,FP,FN

def Cal_Time(file):

    data = pd.read_csv(file)
    labels = data['target'].to_numpy()

    features = data.drop('target', axis=1).to_numpy()
    start = timer()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=45)
    svm_classifier = SVC()

    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)

    end = timer()
    
    return timedelta(seconds=end-start)

def Cal_PeakTime(file):

    data = pd.read_csv(file)
    labels = data['target'].to_numpy()

    features = data.drop('target', axis=1).to_numpy()
    tracemalloc.start()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=45)
    svm_classifier = SVC()

    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)

    current, peak = tracemalloc.get_traced_memory()
    result = peak / (1024 * 1024)
    
    return result

def Cal_CurrentTime(file):

    data = pd.read_csv(file)
    labels = data['target'].to_numpy()

    features = data.drop('target', axis=1).to_numpy()
    tracemalloc.start()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm_classifier = SVC()

    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)

    current, peak = tracemalloc.get_traced_memory()
    result = current / (1024 * 1024)
    
    return result

total_acc=[]
for fileCounter in range(2,3):

    folderName = r'Data/' + str(fileCounter) + '-mix-.csv'
    acc,fs,tp,tn,fp,fn = SVM(folderName,fileCounter)

    total_acc.append(acc)
    
    print('\n________________________________ Result:'+str(fileCounter)+' ____________________________________')
    print("\n==> accuracy:",acc)

    print("\n==> F-Score:",fs)
    print("\n==> TP:",tp)
             
    print("\n==> TN:",tn)
    print("\n==> FP:",fp)

    print("\n==> FN:",fn)
    # print("\n==> Current Memory:",Cal_CurrentTime(folderName))

    # print("\n==> Peak Memory:",Cal_PeakTime(folderName))
    # print("\n==> Times:",Cal_Time(folderName))

    # print('\n________________________________________________________________________________________')

