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



# (3)maximum accuracies模擬暫時放遺下
k_values = np.linspace(0.4, 0.6, 101)
accuracies = []

for k in k_values:
    y_pred = (clf.predict_proba(X_test)[:,1] >= k).astype(bool)

    accuracies.append(accuracy_score(y_test, y_pred))

max_accuracy_index = np.argmax(accuracies)
max_accuracy_k = k_values[max_accuracy_index]
max_accuracy = accuracies[max_accuracy_index]

plt.figure()
plt.plot(k_values, accuracies, label='Mean Accuracies')  # add label
plt.plot(max_accuracy_k, max_accuracy, 'ro', label='Max Mean Accuracies')  # add label
plt.annotate(f'({max_accuracy_k:.2f}, {max_accuracy:.2f})', (max_accuracy_k, max_accuracy), textcoords="offset points", xytext=(-10,-10), ha='center')  # annotate the coordinate
plt.xlabel('Threshold Values k')
plt.ylabel('Accuracy')
plt.title('Mean Accuracies for Different Threshold Values')
plt.legend()  # show legend
plt.savefig('Mean Accuracies for Different Threshold Values.png')


k_values = np.linspace(0, 1, 101)
mean_accuracies = []

for k in k_values:
    accuracies_k = []
    for clf, X_test, y_test in zip(clf, X_test, y_test):
        y_pred = (clf.predict_proba(X_test)[:, 1] >= k).astype(int)
        accuracies_k.append(accuracy_score(y_test, y_pred))
    mean_accuracies.append(np.mean(accuracies_k))

# 找到最大平均精度對應的k值
max_mean_accuracy = max(mean_accuracies)
optimal_k_for_max_mean_accuracy = k_values[mean_accuracies.index(max_mean_accuracy)]

# 繪製曲線圖
plt.figure(figsize=(8, 5))
plt.plot(k_values, mean_accuracies, label='Mean Accuracies')
plt.scatter([optimal_k_for_max_mean_accuracy], [max_mean_accuracy], color='red')  # 高亮最大平均精度
plt.title('Mean Accuracies for Different Threshold values')
plt.xlabel('Threshold Values k')
plt.ylabel('Mean Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()