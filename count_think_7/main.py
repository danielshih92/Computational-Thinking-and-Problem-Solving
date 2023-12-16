# read oddExperiment.txt

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, r2_score
# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
# 讀取數據
def read_xy_from_file(file_path):
    """
    This function reads x and y values from a file and returns them as two lists.
    The file is expected to have one pair of x and y per line, separated by a space or comma.
    """
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    with open(file_path, 'r') as file:
        lines = file.readlines()
        x_values = []
        y_values = []
        for line in lines:
            values = [value for value in line.split() if value.strip() and is_float(value)]
            if len(values) == 2:
                x, y = map(float, values)
                x_values.append(x)
                y_values.append(y)
        return y_values, x_values

def plot1(x, y):
    # Linear fit
    linear_coeffs = np.polyfit(x, y, 1)
    linear_fit = np.polyval(linear_coeffs, x)

    # Quadratic fit
    quad_coeffs = np.polyfit(x, y, 2)
    quad_fit = np.polyval(quad_coeffs, x)

    # Calculate and print the error metrics
    linear_mse = mean_squared_error(y, linear_fit)
    linear_r2 = r2_score(y, linear_fit)

    quad_mse = mean_squared_error(y, quad_fit)
    quad_r2 = r2_score(y, quad_fit)

    # print("Linear fit:")
    # print("Mean squared error: ", linear_mse)
    # print("R^2: ", linear_r2)

    # print("Quadratic fit:")
    # print("Mean squared error: ", quad_mse)
    # print("R^2: ", quad_r2)

    # Plotting the data and fits
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    plt.plot(x, linear_fit, label=f'Fit of degree 1, LSE = {linear_mse:.2f}, R² = {linear_r2:.5f}', color='green')
    plt.plot(x, quad_fit, label=f'Fit of degree 2, LSE = {quad_mse:.2f}, R² = {quad_r2:.5f}', color='red')
    plt.title('OddExperiment Data')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), fancybox=True, shadow=True, ncol=1)
    plt.grid(True)
    plt.savefig('OddExperiment.png')
    plt.show()

def plot2(x, y):
    # digree `1`
    linear_coeffs = np.polyfit(x, y, 1)
    linear_fit = np.polyval(linear_coeffs, x)

    # digree `2`
    quad_coeffs = np.polyfit(x, y, 2)
    quad_fit = np.polyval(quad_coeffs, x)

    # digree `4`
    four_coeffs = np.polyfit(x, y, 4)
    four_fit = np.polyval(four_coeffs, x)

    # digree `8`
    eight_coeffs = np.polyfit(x, y, 8)
    eight_fit = np.polyval(eight_coeffs, x)

    # digree `16`
    sixteen_coeffs = np.polyfit(x, y, 16)
    sixteen_fit = np.polyval(sixteen_coeffs, x)

    # Calculate and print the error metrics
    linear_mse = mean_squared_error(y, linear_fit)
    linear_r2 = r2_score(y, linear_fit)

    quad_mse = mean_squared_error(y, quad_fit)
    quad_r2 = r2_score(y, quad_fit)

    four_mse = mean_squared_error(y, four_fit)
    four_r2 = r2_score(y, four_fit)

    eight_mse = mean_squared_error(y, eight_fit)
    eight_r2 = r2_score(y, eight_fit)

    sixteen_mse = mean_squared_error(y, sixteen_fit)
    sixteen_r2 = r2_score(y, sixteen_fit)

     # Plotting the data and fits
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    plt.plot(x, linear_fit, label=f'Fit of degree 1, R² = {linear_r2:.5f}', color='green')
    plt.plot(x, quad_fit, label=f'Fit of degree 2, R² = {quad_r2:.5f}', color='red')
    plt.plot(x, four_fit, label=f'Fit of degree 4, R² = {four_r2:.5f}', color='blue')
    plt.plot(x, eight_fit, label=f'Fit of degree 8, R² = {eight_r2:.5f}', color='orange')
    plt.plot(x, sixteen_fit, label=f'Fit of degree 16, R² = {sixteen_r2:.5f}', color='purple')
    plt.title('OddExperiment Data')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), fancybox=True, shadow=True, ncol=1)
    plt.grid(True)
    plt.savefig('OddExperiment2.png')
    plt.show()
    return quad_coeffs, quad_fit, sixteen_coeffs, sixteen_fit
def compare(x,y, a, b, c, d):
    # digree `2`
    quad_coeffs = a
    quad_fit = b

    # digree `16`
    sixteen_coeffs = c
    sixteen_fit = d

    quad_r2 = r2_score(y, quad_fit)
    sixteen_r2 = r2_score(y, sixteen_fit)

    # Plotting the data and fits
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='test_Data')
    plt.plot(x, quad_fit, label=f'Fit of degree 2 of oddExperiment, R² = {quad_r2:.5f}', color='red')
    plt.plot(x, sixteen_fit, label=f'Fit of degree 16 of oddExperiment, R² = {sixteen_r2:.5f}', color='purple')
    plt.title('Test Data')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), fancybox=True, shadow=True, ncol=1)
    plt.grid(True)
    plt.savefig('Test_Data.png')
    plt.show()
# main code
path = 'oddExperiment.txt'
path2 = 'TestDataSet.txt'
x, y = read_xy_from_file(path)
test_x, test_y = read_xy_from_file(path2)
plot1(x, y)
a, b, c, d = plot2(x, y)
compare(test_x, test_y,a, b, c, d)