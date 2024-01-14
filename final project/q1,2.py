import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, survived_data, bins, xlabel, ylabel, title, filename):
    data_mean = np.mean(data)
    data_std = np.std(data)
    survived_data_mean = np.mean(survived_data)
    survived_data_std = np.std(survived_data)
    all_label = 'All Passengers\nMean = {:.2f} SD = {:.2f}'.format(data_mean, data_std)
    survived_label = 'Survived Passengers\nMean = {:.2f} SD = {:.2f}'.format(survived_data_mean, survived_data_std)

    plt.figure()
    plt.hist(data, bins=bins, label=all_label, edgecolor='black')
    plt.hist(survived_data, bins=bins, label=survived_label, edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(filename)

def process_data():
    male_class = []
    male_age = []
    male_survived = []
    female_class = []
    female_age = []
    female_survived = []
    with open('TitanicPassengers.txt', 'r') as file:
        next(file)
        lines = file.readlines()
        for line in lines:
            val = line.split(',')
            if(val[2] == 'M'):
                male_class.append(int(val[0]))
                male_age.append(int(float(val[1])))
                male_survived.append(int(val[3]))
            else:
                female_class.append(int(val[0]))
                female_age.append(int(float(val[1])))
                female_survived.append(int(val[3]))

    male_survived_age = [male_age[i] for i in range(len(male_age)) if male_survived[i] == 1]
    female_survived_age = [female_age[i] for i in range(len(female_age)) if female_survived[i] == 1]
    male_survived_class = [male_class[i] for i in range(len(male_class)) if male_survived[i] == 1]
    female_survived_class = [female_class[i] for i in range(len(female_class)) if female_survived[i] == 1]

    plot_histogram(male_age, male_survived_age, 20, 'Male Ages', 'Number of Male Passengers', 'Male Passengers and Survived', 'Male Passengers and Survived.png')
    plot_histogram(female_age, female_survived_age, 20, 'Female Ages', 'Number of Female Passengers', 'Female Passengers and Survived', 'Female Passengers and Survived.png')
    plot_histogram(male_class, male_survived_class, 3, 'Class', 'Number of Male Passengers', 'Male Cabin Classes and Survived', 'Male Cabin Classes and Survived.png')
    plot_histogram(female_class, female_survived_class, 3, 'Class', 'Number of Female Passengers', 'Female Cabin Classes and Survived', 'Female Cabin Classes and Survived.png')

# main 
process_data()




