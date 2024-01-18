import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def accuracy(truePos, falsePos, trueNeg, falseNeg): 
    numerator = truePos + trueNeg 
    denominator = truePos + trueNeg + falsePos + falseNeg 
    return numerator/denominator 
def sensitivity(truePos, falseNeg): 
    try: 
        return truePos/(truePos + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 
def specificity(trueNeg, falsePos): 
    try: 
        return trueNeg/(trueNeg + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def posPredVal(truePos, falsePos): 
    try:
        return truePos/(truePos + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
    
def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True): 
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
    sens = sensitivity(truePos, falseNeg) 
    spec = specificity(trueNeg, falsePos) 
    ppv = posPredVal(truePos, falsePos) 
    if toPrint: 
        print(' Accuracy =', round(accur, 3)) 
        print(' Sensitivity =', round(sens, 3)) 
        print(' Specificity =', round(spec, 3)) 
        print(' Pos. Pred. Val. =', round(ppv, 3)) 
    return (accur, sens, spec, ppv) 

def confusionMatrix(truePos, falsePos, trueNeg, falseNeg):
#    print('\nk = ', k)
    print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
    print('                     ','TP', ' \t', 'FP' )
    print('Confusion Matrix is: ', truePos, '\t', falsePos)
    print('                     ', trueNeg, '\t', falseNeg)
    print('                     ', 'TN', ' \t', 'FN' )    
    getStats(truePos, falsePos, trueNeg, falseNeg)
    return

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Count the labels
        label_counts = {}
        for label in k_nearest_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        # return the label with the most count
        most_common_label = max(label_counts, key=label_counts.get)
        return most_common_label
    def get_params(self, deep=True):
        return {"k": self.k}

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


X = np.array([pas_class, pas_age, pas_gender]).T
y = np.array(pas_survived)

# Split the data into male and female
X_male = X[X[:, 2] == 1]
y_male = y[X[:, 2] == 1]
X_female = X[X[:, 2] == 0]
y_female = y[X[:, 2] == 0]

# Set k to 3
k = 3

# Train and predict for male
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2)
knn_male = KNN(k=k)
knn_male.fit(X_train_male, y_train_male)
y_pred_male = knn_male.predict(X_test_male)

# Compute the confusion matrix for male
C_M_male = confusion_matrix(y_test_male, y_pred_male)
print("Try to predict male and female separately and combined with k=3:\n")
print("For male:")

trueNeg_male = C_M_male[0][0]
falsePos_male = C_M_male[0][1]
falseNeg_male = C_M_male[1][0]
truePos_male = C_M_male[1][1]

confusionMatrix(truePos_male, falsePos_male, trueNeg_male, falseNeg_male)

# Train and predict for female
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2)
knn_female = KNN(k=k)
knn_female.fit(X_train_female, y_train_female)
y_pred_female = knn_female.predict(X_test_female)

# Compute the confusion matrix for female
C_M_female = confusion_matrix(y_test_female, y_pred_female)
print("\nFor female:")

trueNeg_female = C_M_female[0][0]
falsePos_female = C_M_female[0][1]
falseNeg_female = C_M_female[1][0]
truePos_female = C_M_female[1][1]

confusionMatrix(truePos_female, falsePos_female, trueNeg_female, falseNeg_female)
# combined
print("\nCombined Predictions Statistics:")
truePos_combined = truePos_female + truePos_male
falsePos_combined = falsePos_female + falsePos_male
trueNeg_combined = trueNeg_female + trueNeg_male
falseNeg_combined = falseNeg_female + falseNeg_male
confusionMatrix(truePos_combined, falsePos_combined, trueNeg_combined, falseNeg_combined)