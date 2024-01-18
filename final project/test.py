# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix

# def accuracy(truePos, falsePos, trueNeg, falseNeg): 
#     numerator = truePos + trueNeg 
#     denominator = truePos + trueNeg + falsePos + falseNeg 
#     return numerator/denominator 
# def sensitivity(truePos, falseNeg): 
#     try: 
#         return truePos/(truePos + falseNeg) 
#     except ZeroDivisionError: 
#         return float('nan') 
# def specificity(trueNeg, falsePos): 
#     try: 
#         return trueNeg/(trueNeg + falsePos) 
#     except ZeroDivisionError: 
#         return float('nan') 
# def posPredVal(truePos, falsePos): 
#     try:
#         return truePos/(truePos + falsePos) 
#     except ZeroDivisionError: 
#         return float('nan') 
    
# def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True): 
#     accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
#     sens = sensitivity(truePos, falseNeg) 
#     spec = specificity(trueNeg, falsePos) 
#     ppv = posPredVal(truePos, falsePos) 
#     if toPrint: 
#         print(' Accuracy =', round(accur, 3)) 
#         print(' Sensitivity =', round(sens, 3)) 
#         print(' Specificity =', round(spec, 3)) 
#         print(' Pos. Pred. Val. =', round(ppv, 3)) 
#     return (accur, sens, spec, ppv) 

# def confusionMatrix(truePos, falsePos, trueNeg, falseNeg):
# #    print('\nk = ', k)
#     print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
#     print('                     ','TP', ' \t', 'FP' )
#     print('Confusion Matrix is: ', truePos, '\t', falsePos)
#     print('                     ', trueNeg, '\t', falseNeg)
#     print('                     ', 'TN', ' \t', 'FN' )    
#     getStats(truePos, falsePos, trueNeg, falseNeg)
#     return

# def plotHist(k_values, cv_scores, real_pred_scores):
#     plt.figure(figsize=(10, 6))
#     plt.plot(k_values, cv_scores, label='n-fold cross validation')
#     plt.plot(k_values, real_pred_scores, label='Real Prediction')
#     plt.xlabel('k values for KNN Regression')
#     plt.ylabel('Accuracy')
#     plt.xticks(range(1, 26, 2))
#     plt.title('Average Accuracy vs k (10 folds)')
#     plt.legend()
#     plt.savefig('Average Accuracy vs k (10 folds).png')
# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2)**2))

# class KNN:
#     def __init__(self, k=3):
#         self.k = k

#     def fit(self, X, y):
#         self.X_train = X
#         self.y_train = y

#     def predict(self, X):
#         y_pred = [self._predict(x) for x in X]
#         return np.array(y_pred)

#     def _predict(self, x):
#         # Compute distances between x and all examples in the training set
#         distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
#         # Sort by distance and return indices of the first k neighbors
#         k_indices = np.argsort(distances)[:self.k]
#         # Extract the labels of the k nearest neighbor training samples
#         k_nearest_labels = [self.y_train[i] for i in k_indices]
#         # Count the labels
#         label_counts = {}
#         for label in k_nearest_labels:
#             if label not in label_counts:
#                 label_counts[label] = 0
#             label_counts[label] += 1
#         # return the label with the most count
#         most_common_label = max(label_counts, key=label_counts.get)
#         return most_common_label
#     def get_params(self, deep=True):
#         return {"k": self.k}

# pas_class = []
# pas_age = []
# pas_survived = []
# pas_gender = []
# with open('TitanicPassengers.txt', 'r') as file:
#     next(file)
#     lines = file.readlines()
#     for line in lines:
#         val = line.split(',')
#         pas_class.append(int(val[0]))
#         pas_age.append(float(val[1]))
#         pas_gender.append(1 if val[2] == 'M' else 0)
#         pas_survived.append(int(val[3]))


# X = np.array([pas_class, pas_age, pas_gender]).T
# y = np.array(pas_survived)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# knn = KNN(k=3)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# C_M = confusion_matrix(y_test, y_pred)
# trueNeg = C_M[0][0]
# falsePos = C_M[0][1]
# falseNeg = C_M[1][0]
# truePos = C_M[1][1]
# print("k-NN Prediction for Survive with k=3:")

# confusionMatrix(truePos, falsePos, trueNeg, falseNeg)

# k_values = list(range(1, 26, 2))
# cv_scores = []
# test_scores = []

# for k in k_values:
#     knn = KNN(k=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     cv_scores.append(scores.mean())
    
#     # Fit the model on the training data and evaluate it on the test data
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     test_score = accuracy_score(y_test, y_pred)
#     test_scores.append(test_score)

# optimal_k = k_values[test_scores.index(max(test_scores))]
# print("K for Maximum Accuracy is:", optimal_k)

# knn = KNN(k=optimal_k)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# C_M = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix for optimal k:")

# trueNeg = C_M[0][0]
# falsePos = C_M[0][1]
# falseNeg = C_M[1][0]
# truePos = C_M[1][1]

# confusionMatrix(truePos, falsePos, trueNeg, falseNeg)
# print("Predictions with maximum accuracy k:", optimal_k)
# print("Cross Validation Accuracies is:", cv_scores[k_values.index(optimal_k)])
# print("Predicted Accuracies is:", test_scores[k_values.index(optimal_k)])
# plotHist(k_values, cv_scores, test_scores)