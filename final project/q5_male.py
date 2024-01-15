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
    for k in np.linspace(0, 1, 100):
        y_pred = (clf.predict_proba(X)[:,1] >= k).astype(int)
        accuracy = accuracy_score(y, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_k = k
    return optimal_k, max_accuracy

# main code
# read data
pas_class_male = []
pas_age_male = []
pas_survived_male = []

pas_gender = []

pas_class_female = []
pas_age_female = []
pas_survived_female = []

with open('TitanicPassengers.txt', 'r') as file:
    next(file)
    lines = file.readlines()
    for line in lines:
        val = line.split(',')
        if val[2] == 'M':
            pas_class_male.append(int(val[0]))
            pas_age_male.append(float(val[1]))
            pas_survived_male.append(int(val[3]))
        else:
            pas_class_female.append(int(val[0]))
            pas_age_female.append(float(val[1]))
            pas_survived_female.append(int(val[3]))

enc = OneHotEncoder()
pas_class_onehot_male = enc.fit_transform(np.array(pas_class_male).reshape(-1, 1)).toarray()
pas_class_onehot_female = enc.fit_transform(np.array(pas_class_female).reshape(-1, 1)).toarray()

X_male = np.hstack((pas_class_onehot_male, np.array(pas_age_male).reshape(-1, 1)))
y_male = np.array(pas_survived_male)

X_female = np.hstack((pas_class_onehot_female, np.array(pas_age_female).reshape(-1, 1)))
y_female = np.array(pas_survived_female)


weights = []
accuracies = []
sensitivities = []
specificities = []
ppv = []
auroc = []
optimal_ks = []
max_accuracies = []
accuracies_for_k = {k: [] for k in np.linspace(0, 1, 100)}

# Build the model and simulate 1000 trials
for _ in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X_male, y_male, test_size=0.2)

    # 建立模型
    # Perform logistic regression for males
    clf_male = LogisticRegression(max_iter=1000)
    clf_male.fit(X_male, y_male)
    # clf_female = LogisticRegression(max_iter=1000)
    # clf_female.fit(X_female, y_female)

    weights.append(clf_male.coef_[0])   

    y_pred = clf_male.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracies.append(accuracy_score(y_test, y_pred))
    sensitivities.append(tp / (tp + fn))
    specificities.append(tn / (tn + fp))
    auroc.append(roc_auc_score(y_test, y_pred))
    if (tp + fp) != 0:
        ppv.append(tp / (tp + fp))
    else:
        ppv.append(np.nan)

    # 第二張圖
    # 尋找最優閾值k
    optimal_k, max_accuracy = find_optimal_threshold(clf_male, X_test, y_test)
    
    # 收集數據
    optimal_ks.append(optimal_k)
    max_accuracies.append(max_accuracy)
    # 第三張圖
    for k in np.linspace(0, 1, 100):
        y_pred = (clf_male.predict_proba(X_test)[:,1] >= k).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies_for_k[k].append(accuracy)

weights = np.array(weights)

mean_weights = np.mean(weights, axis=0)
lower_weights, upper_weights = np.percentile(weights, [2.5, 97.5], axis=0)

mean_accuracy = np.mean(accuracies)
lower_accuracy, upper_accuracy = np.percentile(accuracies, [2.5, 97.5])

std_max_accuracy = np.std(max_accuracies)

mean_sensitivity = np.mean(sensitivities)
lower_sensitivity, upper_sensitivity = np.percentile(sensitivities, [2.5, 97.5])

mean_specificity = np.mean(specificities)
lower_specificity, upper_specificity = np.percentile(specificities, [2.5, 97.5])

mean_ppv = np.mean(ppv)
lower_ppv, upper_ppv = np.percentile(ppv, [2.5, 97.5])

mean_auroc = np.mean(auroc)
lower_auroc, upper_auroc = np.percentile(auroc, [2.5, 97.5])

print('Logistic Regression:')
print('Averages for Male examples 1000 trials with k=0.5')
print('Mean weight of C1 = {}, 95% confidence interval = {}'.format(round(mean_weights[0], 3), round(upper_weights[0] - lower_weights[0], 3)))
print('Mean weight of C2 = {}, 95% confidence interval = {}'.format(round(mean_weights[1], 3), round(upper_weights[1] - lower_weights[1], 3)))
print('Mean weight of C3 = {}, 95% confidence interval = {}'.format(round(mean_weights[2], 3), round(upper_weights[2] - lower_weights[2], 3)))
print('Mean weight of age = {},  95% confidence interval = {}'.format(round(mean_weights[3], 3), round(upper_weights[3] - lower_weights[3], 3)))
print('Mean weight of Male Gender = 0.0, 95% CI = 0.0')
print('Mean accuracy = {},  95% confidence interval = {}'.format(round(mean_accuracy, 3), round(upper_accuracy - lower_accuracy, 3)))
print('Mean sensitivity = {},  95% confidence interval = {}'.format(round(mean_sensitivity, 3), round(upper_sensitivity - lower_sensitivity, 3)))
print('Mean specificity = {},  95% confidence interval = {}'.format(round(mean_specificity, 3), round(upper_specificity - lower_specificity, 3)))
print('Mean pos. pred. val. = {},  95% confidence interval = {}'.format(round(mean_ppv, 3), round(upper_ppv - lower_ppv, 3)))
print('Mean AUROC = {},  95% confidence interval = {}'.format(round(mean_auroc, 3), round(upper_auroc - lower_auroc, 3)))

# (1)maxium accuracies
# Calculate the mean and standard deviation of max_accuracies
plt.figure()
plt.hist(accuracies, bins=20, edgecolor='black', label='Mean Accuracies\nMean = {:.2f} SD = {:.2f}'.format(mean_accuracy, std_max_accuracy))
plt.xlabel('Maximum Accuracies')
plt.ylabel('Numbers of Maximum Accuracies')
plt.title('Male: Maximum Accuracies')
plt.legend(loc='upper left')
plt.savefig('(Male)Maximum Accuracies.png')

# (2)threshold values k for maximum accuracies
# 計算k值和最大精度的平均值和標準差
mean_optimal_k = np.mean(optimal_ks)
std_optimal_k = np.std(optimal_ks)
mean_max_accuracy = np.mean(max_accuracies)
std_max_accuracy = np.std(max_accuracies)

# Plot the histogram for optimal threshold values k
plt.figure(figsize=(10, 6))
plt.hist(optimal_ks, bins=20, range=(0.4, 0.6), edgecolor='black', label='k values for maximum accuracies\nMean ={:.2f} SD = {:.2f}'.format(mean_optimal_k, std_optimal_k))
plt.title('Male: Threshold Value k for Maximum Accuracies')
plt.xlabel('Threshold Values k')
plt.ylabel('Number of ks')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('(Male)Threshold values k for Maximum Accuracies.png')

# (3)mean accuracies for different threshold values
# 計算每個閾值的平均準確度
mean_accuracies = {k: np.mean(v) for k, v in accuracies_for_k.items()}

# 繪製平均準確度
plt.figure(figsize=(10, 6))
plt.plot(list(mean_accuracies.keys()), list(mean_accuracies.values()), label='Mean Accuracies')
max_accuracy_k = max(mean_accuracies, key=mean_accuracies.get)
plt.scatter(max_accuracy_k, mean_accuracies[max_accuracy_k], label='Maximum Mean Accuracy', color='red')  # 最大準確度的點
plt.text(max_accuracy_k, mean_accuracies[max_accuracy_k], '({:.3f}, {:.3f})'.format(max_accuracy_k, mean_accuracies[max_accuracy_k]), ha='right') 
plt.title('Male: Mean accuracies for different threshold values')
plt.xlabel('Threshold values')
plt.ylabel('Mean Accuracies')
plt.xlim(0.4, 0.6)
plt.ylim(0.7750, 0.7925)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('(Male)Mean accuracies for different threshold values.png')
