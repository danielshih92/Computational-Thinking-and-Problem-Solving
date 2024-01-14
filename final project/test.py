import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

def find_optimal_threshold(clf, X, y):
    optimal_k = 0.5
    max_accuracy = 0
    
    # 考慮閾值k從0到1的100個值
    for k in np.linspace(0, 1, 101):
        y_pred = (clf.predict_proba(X)[:,1] >= k).astype(int)
        accuracy = accuracy_score(y, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_k = k
    return optimal_k, max_accuracy

pas_class = []
pas_age = []
pas_survived = []
pas_gender = []
with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        pas_class.append(int(val[0]))
        pas_age.append(float(val[1]))
        pas_gender.append(1 if val[2] == 'M' else 0)
        pas_survived.append(int(val[3]))

weights = []
accuracies = []
sensitivities = []
specificities = []
ppv = []
auroc = []
optimal_ks = []
max_accuracies = []

for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 建立模型
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    weights.append(clf.coef_[0])

    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracies.append(accuracy_score(y_test, y_pred))
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))
    ppv.append(tp / (tp + fp))
    auroc.append(roc_auc_score(y_test, y_pred))

    # 尋找最優閾值k
    optimal_k, max_accuracy = find_optimal_threshold(clf, X_test, y_test)
    
    # 收集數據
    optimal_ks.append(optimal_k)
    max_accuracies.append(max_accuracy)

# (2)threshold values k for maximum accuracies
# 計算k值和最大精度的平均值和標準差
mean_optimal_k = np.mean(optimal_ks)
std_optimal_k = np.std(optimal_ks)
mean_max_accuracy = np.mean(max_accuracies)
std_max_accuracy = np.std(max_accuracies)

# Plot the histogram for optimal threshold values k
plt.figure(figsize=(10, 6))
plt.hist(optimal_ks, bins=20, color='blue', edgecolor='black')
plt.title('Optimal Threshold Values k for Maximum Accuracies')
plt.xlabel('Optimal Threshold Values k')
plt.ylabel('Frequency')
# Add text for mean and standard deviation
plt.text(mean_optimal_k + 0.03, plt.ylim()[1] * 0.9, f'Mean = {mean_optimal_k:.2f}\nSD = {std_optimal_k:.2f}',
         horizontalalignment='center', color='red')
plt.tight_layout()
plt.savefig('Threshold values k for Maximum Accuracies.png')