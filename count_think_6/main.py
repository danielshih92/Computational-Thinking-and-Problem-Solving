import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import random
import time


def read_scores_from_file(file_path):
    """
    This function reads test scores from a file and returns them as a list of floats.
    The file is expected to have one score per line or multiple scores in one line separated by commas.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        scores = []
        for line in lines:
            scores.extend([float(score) for score in line.split(',') if score.strip()])
        return scores


def plot_histogram(scores, title='Test Scores', xlabel='Scores', ylabel='Number of Students'):
    """
    This function takes a list of scores and plots a histogram.
    """
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)

    # Annotate the mean and standard deviation on the plot
    plt.text(min(scores), max(np.histogram(scores, bins=30)[0]) * 0.9,
             f'Mean = {mean_score:.2f}\nSD = {std_deviation:.2f}',
             bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(title + '.png')
    plt.show()


def plot_normal_distribution(scores):
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)
    the_scores = np.array(scores)

    # Calculate confidence interval
    lower_bound = mean_score - 1.96 * std_deviation
    upper_bound = mean_score + 1.96 * std_deviation

    # Calculate the required numbers
    num_above_60 = np.sum(the_scores > 60)
    percent_above_60 = num_above_60 / len(scores) * 100

    num_in_confidence_interval = np.sum((the_scores >= lower_bound) & (the_scores <= upper_bound))

    num_below_lower_bound = np.sum(the_scores < lower_bound)
    num_above_upper_bound = np.sum(the_scores > upper_bound)

    # Print the information
    print(f"Real number of scores above 60 is: {num_above_60}, around {percent_above_60:.2f}%")
    print(f"Precise 95% number of scores between mu-1.96*sigma and mu+1.96*sigma is: {num_in_confidence_interval}")
    print(f"Number of scores below mu-1.96*sigma ({lower_bound:.2f}) is: {num_below_lower_bound}")
    print(f"Number of scores above mu+1.96*sigma ({upper_bound:.2f}) is: {num_above_upper_bound}")

    # Plot the normal distribution
    x_values = np.linspace(mean_score - 4 * std_deviation, mean_score + 4 * std_deviation, 1000)
    y_values = stats.norm.pdf(x_values, mean_score, std_deviation)
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, color='blue')

    y_at_mean = stats.norm.pdf(mean_score, mean_score, std_deviation)
    y_at_lower = stats.norm.pdf(lower_bound, mean_score, std_deviation)
    y_at_upper = stats.norm.pdf(upper_bound, mean_score, std_deviation)

    # Plot the mean and confidence interval
    plt.plot([mean_score, mean_score], [0, y_at_mean], color='black', linestyle='-', linewidth=2)
    plt.plot([lower_bound, lower_bound], [0, y_at_lower], color='black', linestyle='--')
    plt.plot([upper_bound, upper_bound], [0, y_at_upper], color='black', linestyle='--')

    # Some text to explain the plot
    plt.title('Normal Distribution: μ = {:.2f}, σ = {:.2f}'.format(mean_score, std_deviation))
    plt.xlabel('Test Scores')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('Normal Distribution.png')
    plt.show()


def draw_estimate_and_sample_normal_distribution(scores, sample_sizes):
    mean = np.mean(scores)
    std_dev = np.std(scores)
    sample_means = []
    sample_std_devs = []
    standard_errors = []
    estimated_standard_errors = []

    random.seed(time.time())

    for size in sample_sizes:
        sample = random.sample(scores, size)
        sample_mean = np.mean(sample)
        sample_std_dev = np.std(sample)
        sample_means.append(sample_mean)
        sample_std_devs.append(sample_std_dev)
        standard_errors.append(std_dev / np.sqrt(size))
        estimated_standard_errors.append(sample_std_dev / np.sqrt(size))

    move = 10
    standard_error_percentages = [error / mean * 100 * 2 for error in standard_errors]
    estimated_standard_error_percentages = [error / mean * 100 * 2 for error in estimated_standard_errors]
    plt.figure()
    x_values = sample_sizes
    y_values = [error / sample_mean * 100 + sample_mean for error, sample_mean in zip(standard_errors, sample_means)]
    for x, y, percentage in zip(x_values, y_values, standard_error_percentages):
        plt.text(x, y, str(round(percentage, 2)) + '%', color='purple')
    y_values = [- error / sample_mean * 100 + sample_mean for error, sample_mean in
                zip(estimated_standard_errors, sample_means)]
    for x, y, percentage in zip(x_values, y_values, estimated_standard_error_percentages):
        plt.text(x, y, str(round(percentage, 2)) + '%', color='green')
    plt.errorbar([size - move for size in sample_sizes], sample_means, yerr=standard_errors, fmt='o',
                 label='95% SE confidence interval')
    plt.errorbar(sample_sizes, sample_means, yerr=estimated_standard_errors, fmt='o-',
                 label='95% Estimated SE confidence interval')
    plt.axhline(y=mean, color='blue', linestyle='--', label='Population mean')
    plt.title('Estimated of Mean Scores')
    plt.xlabel('Sample Size')
    plt.ylabel('Scores')
    plt.xticks(np.arange(0, 700, 100))
    plt.yticks(np.arange(65, 75, 1))
    plt.legend()
    plt.savefig('Estimated of Mean Scores.png')
    plt.show()
    for i in range(3):
        plt.figure()
        x = np.linspace(min(scores), max(scores), 100)
        y = norm.pdf(x, sample_means[i], sample_std_devs[i])
        plt.plot(x, y, label='Mean={:.2f}, SD={:.2f}'.format(sample_means[i], sample_std_devs[i]))
        plt.title('Normal Distribution for ' + str(sample_sizes[i]) + ' sample size')
        y_mean = norm.pdf(sample_means[i], sample_means[i], sample_std_devs[i])
        plt.plot([sample_means[i], sample_means[i]], [0, y_mean], color='blue')
        y_std_dev1 = norm.pdf(sample_means[i] + standard_errors[0], sample_means[i], sample_std_devs[i])
        y_std_dev2 = norm.pdf(sample_means[i] - standard_errors[0], sample_means[i], sample_std_devs[i])
        plt.plot([sample_means[i] + standard_errors[0], sample_means[i] + standard_errors[0]], [0, y_std_dev1],
                 color='red', linestyle='dashed')
        plt.plot([sample_means[i] - standard_errors[0], sample_means[i] - standard_errors[0]], [0, y_std_dev2],
                 color='red', linestyle='dashed')
        y_mean = norm.pdf(mean, sample_means[i], sample_std_devs[i])
        plt.plot([mean, mean], [0, y_mean], color='black', label='Mean={:.2f}, SD={:.2f}'.format(mean, std_dev))
        plt.legend()
        plt.savefig('Normal Distribution for ' + str(sample_sizes[i]) + ' sample size.png')
        plt.show()
    print('*' * 80)

    for i in range(3):
        print('Normal Distribution for ' + str(sample_sizes[i]) + ' sample size')
        print('Estimated number of scores above 60 is: ' + str(
            int(round(norm.sf(60, sample_means[i], sample_std_devs[i]) * 12500))) + ', around ' + str(
            int(round(norm.sf(60, sample_means[i], sample_std_devs[i]) * 100))) + '%')
        num_scores_within_range = sum(
            sample_means[i] - 1.96 * sample_std_devs[i] <= score <= sample_means[i] + 1.96 * sample_std_devs[i] for
            score in scores)
        print(f'Estimated 95% number of scores between mu-1.96*sigma and mu+1.96*sigma is: {num_scores_within_range}')
        num_scores_below_range = sum(score < sample_means[i] - 1.96 * sample_std_devs[i] for score in scores)
        print(
            f'Number of scores below mu-1.96*sigma({round(sample_means[i] - 1.96 * sample_std_devs[i], 2)}) is: {num_scores_below_range}')
        num_scores_above_range = sum(score > sample_means[i] + 1.96 * sample_std_devs[i] for score in scores)
        print(
            f'Number of scores above mu+1.96*sigma({round(sample_means[i] + 1.96 * sample_std_devs[i], 2)}) is: {num_scores_above_range}')
        if i < 2:
            print('*' * 80)


# main code

file_path = 'TestScorerResult.txt'

test_scores = read_scores_from_file(file_path)
sample_size = len(test_scores)

# Plot 1: Histogram of all scores
plot_histogram(test_scores, title='2012 Test Scores - All 12500', xlabel='Test Scores', ylabel='Number of Students')
# Plot 2: Normal distribution of all scores
plot_normal_distribution(test_scores)
# Plot 3: Estimates of mean scores
# Plot 4, 5, 6: Normal distribution of sample scores
draw_estimate_and_sample_normal_distribution(test_scores, [50, 200, 500])



