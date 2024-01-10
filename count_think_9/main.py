import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# No "sex" column in the dataset
columns_to_use = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
abalone_data = pd.read_csv('abalone.txt', usecols=columns_to_use)

# Splitting the data into test set
train_set, test_set = train_test_split(abalone_data, test_size=0.2, random_state=int(time.time()))

# Separating the features and the target variable for training set
X_train = train_set.drop('Rings', axis=1)
y_train = train_set['Rings']

# Training a kNN model for each k value
k_values = [3, 5, 7, 9, 11] 
models = {}

# Initialize dictionary to store R² values
r2_values = {}
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    models[k] = knn

np.random.seed(int(time.time()))  # for reproducibility
# Plotting the age distribution for all abalones
plt.figure(figsize=(10, 6))
plt.hist(abalone_data['Rings'], bins=20, color='blue', edgecolor='black')
plt.title('All Abalone Ring Sizes (Age) Distribution')
plt.xlabel('Ring Sizes')
plt.ylabel('Number of Abalones')
plt.text(0.9, 0.9, f'Mean = {abalone_data["Rings"].mean():.2f}\nSD = {abalone_data["Rings"].std():.2f}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
plt.show()

# Plotting the age distribution for the test set
plt.figure(figsize=(10, 6))
plt.hist(test_set['Rings'], bins=20, color='blue', edgecolor='black')
plt.title('Test Set Abalone Ring Sizes (Age) Distribution')
plt.xlabel('Ring Sizes')
plt.ylabel('Number of Abalones')
plt.text(0.9, 0.9, f'Mean = {test_set["Rings"].mean():.2f}\nSD = {test_set["Rings"].std():.2f}', 
         horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
plt.show()

# Separating the features and the target variable for test set
X_test = test_set.drop('Rings', axis=1)
y_test = test_set['Rings']

for k, model in models.items():
    # Predicting the training set and calculating R² and RMSD
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    rmsd_train = mean_squared_error(y_train, y_train_pred, squared=False)
    print(f'Pre-Training with Whole Examples Evaluation with k= {k}')
    print(f'Coefficient of Determination: rSquare(R²): {r2_train:.4f}')
    print(f'Root Mean Square Deviation Rmsd: {rmsd_train:.4f}')

    # Predicting the test set and calculating R² and RMSD
    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    rmsd_test = mean_squared_error(y_test, y_test_pred, squared=False)
    print(f'After Trained Testing Using Test Set with k= {k}')
    print(f'Coefficient of Determination: rSquare(R²): {r2_test:.4f}')
    print(f'Root Mean Square Deviation Rmsd: {rmsd_test:.4f}')
    print("**********************************************************************")
    # Calculate absolute and percentage errors
    errors = y_test_pred - y_test
    absolute_errors = np.abs(errors)
    percentage_errors = (errors / y_test) * 100 
    
    # Avoid dividing by zero or very small numbers for percentage error calculation
    y_test_with_min_value = np.where(y_test > 0.1, y_test, 0.1)
    percentage_errors = (errors / y_test_with_min_value) * 100

    # Organize errors by ring sizes
    errors_data = pd.DataFrame({
        'ActualRings': y_test,
        'AbsoluteErrors': absolute_errors,
        'PercentageErrors': percentage_errors
    })
    grouped_errors = errors_data.groupby('ActualRings').mean()

    # 在繪圖前計算絕對誤差和百分比誤差的範圍
    abs_error_range = f"[Errors] {absolute_errors.min():.2f} to {absolute_errors.max():.2f}"
    perc_error_range = f"% errors {percentage_errors.min():.2f} to {percentage_errors.max():.2f}"

    # Plotting with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # First axis for Absolute Errors
    color = 'tab:blue'
    ax1.set_xlabel('Ring Sizes')
    ax1.set_ylabel('Absolute Errors', color=color)
    # 包含範圍信息的標籤
    ax1.plot(grouped_errors.index, grouped_errors['AbsoluteErrors'], color=color, marker='o', label=abs_error_range)
    ax1.tick_params(axis='y', labelcolor=color)

    # Instantiate a second axis that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Percentage Errors', color=color)
    # 包含範圍信息的標籤
    ax2.plot(grouped_errors.index, grouped_errors['PercentageErrors'], color=color, marker='o', label=perc_error_range)
    ax2.tick_params(axis='y', labelcolor=color)

    
    plt.title(f'Predict Rings Absolute and Percentage Errors for k={k}', pad = 20)
    fig.tight_layout()  # To ensure the right y-label is not slightly clipped
    # 使用這個方法將圖例放在圖表之外
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.show()
    #**********************************************************************************************************
    # Plot the histogram of the absolute errors
    plt.figure(figsize=(10, 6))
    plt.hist(absolute_errors, bins=20, color='blue', edgecolor='black')
    plt.title(f'Predicted Rings (Ages) Absolute Errors Distribution for k={k}')
    plt.xlabel('Predicted Ring Sizes Absolute Errors')
    plt.ylabel('Number of Abalones')
    plt.text(0.9, 0.9, f'Mean = {absolute_errors.mean():.2f}\nSD = {absolute_errors.std():.2f}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.show()
    
    # Plot the histogram of the percentage errors
    plt.figure(figsize=(10, 6))
    plt.hist(percentage_errors, bins=20, color='blue', edgecolor='black')
    plt.title(f'Predicted Rings (Ages) Percentage Errors Distribution for k={k}')
    plt.xlabel('Predicted Ring Sizes Error Percentages')
    plt.ylabel('Number of Abalones')
    plt.text(0.9, 0.9, f'Mean = {percentage_errors.mean():.2f}\nSD = {percentage_errors.std():.2f}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
    plt.show()
    
    # Store R² value in the dictionary
    r2_values[k] = r2_test

# Find the maximum R² value and the corresponding k
best_k = max(r2_values, key=r2_values.get)
best_r2 = r2_values[best_k]

print(f"The maximum R² value for trained model is {best_r2:.12f}, happens in k = {best_k}")

