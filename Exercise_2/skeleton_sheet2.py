import numpy as np
import matplotlib.pyplot as plt


def mean(x):
    """Calculate the mean for an array-like object x.

    Parameters
    ----------
    x : array-like
        Array-like object containing the data.

    Returns
    -------
    mean : float
        The mean of the data.
    """
    # here goes your code
    sum_x = 0
    for element in x:
        sum_x += element

    mean = sum_x / len(x)
    return mean


def std(x):
    """Calculate the standard deviation for an array-like object x."""
    # here goes your code
    mean_x = mean(x)
    sum_diff_x_mean = 0

    for element in x:
        sum_diff_x_mean += (element - mean_x) ** 2

    variance = sum_diff_x_mean / len(x)
    return np.sqrt(variance)


def variance(x):
    """Calculate the variance for an array-like object x."""
    # here goes your code
    mean_x = mean(x)
    sum_diff_x_mean = 0

    for element in x:
        sum_diff_x_mean += (element - mean_x) ** 2

    variance = sum_diff_x_mean / len(x)
    return variance


def mean_uncertainty(x):
    """Calculate the uncertainty in the mean for an array-like object x."""
    # here goes your code
    std_x = std(x)
    mean_uncertainty = std_x / np.sqrt(len(x))
    return mean_uncertainty


def bin_uncertainty(x, nr_bins):
    """Calculate the uncertainty in the bin for an array-like object x."""
    # Divide x into bins
    bins = np.array_split(x, nr_bins)

    # Calculate the mean of each bin
    bin_uncertainty = []
    for bin in bins:
        bin_uncertainty.append(mean_uncertainty(bin))

    return bin_uncertainty


def covariance(x, y):
    """Calculate the covariance between two array-like objects x and y."""
    mean_x = mean(x)
    mean_y = mean(y)
    sum_diff_x_y = 0

    for i in range(len(x)):
        sum_diff_x_y += (x[i] - mean_x) * (y[i] - mean_y)

    covariance = sum_diff_x_y / len(x)
    return covariance


def correlation(x, y):
    """Calculate the correlation between two array-like objects x and y."""
    # here goes your code
    cov = covariance(x, y)
    std_x = std(x)
    std_y = std(y)

    correlation = cov / (std_x * std_y)
    return correlation


def ex1():
    data = np.loadtxt("ironman.txt")
    age = 2010 - data[:, 1]
    total_time = data[:, 2]

    # a)
    mean_age = mean(age)
    mean_age_uncertainty = mean_uncertainty(age)
    variance_age = variance(age)
    std_age = std(age)

    print(f"The mean age of the participants is {mean_age:.2f} +/- {mean_age_uncertainty:.2f} years.")
    print(f"The variance of the age of the participants is {variance_age:.2f} years.")
    print(f"The standard deviation of the age of the participants is {std_age:.2f} years.")
    print("\n")

    mean_total_time = mean(total_time)
    mean_total_time_uncertainty = mean_uncertainty(total_time)
    variance_total_time = variance(total_time)
    std_total_time = std(total_time)

    print(f"The mean total time of the participants is {mean_total_time:.2f} +/- {mean_total_time_uncertainty:.2f} minutes.")
    print(f"The variance of the total time of the participants is {variance_total_time:.2f} minutes.")
    print(f"The standard deviation of the total time of the participants is {std_total_time:.2f} minutes.")
    print("\n")

    # b)
    data_younger_than_35 = data[age < 35]
    data_older_than_35 = data[age >= 35]

    mean_total_time_younger_than_35 = mean(data_younger_than_35[:, 2])
    mean_total_time_uncertainty_younger_than_35 = mean_uncertainty(data_younger_than_35[:, 2])
    mean_total_time_older_than_35 = mean(data_older_than_35[:, 2])
    mean_total_time_uncertainty_older_than_35 = mean_uncertainty(data_older_than_35[:, 2])

    print(f"The mean total time of the participants younger than 35 is {mean_total_time_younger_than_35:.2f} +/- {mean_total_time_uncertainty_younger_than_35:.2f} minutes.")
    print(f"The mean total time of the participants older than 35 is {mean_total_time_older_than_35:.2f} +/- {mean_total_time_uncertainty_older_than_35:.2f} minutes.")
    print("\n")

    # c)
    # TODO: Implement own algorithm for histogram using plt.bar and compute bins yourself
    # Histogram of Age
    bin_uncertainties = bin_uncertainty(age, 20)

    # d)
    # TODO

    # e)
    # Setup data
    total_rank = data[:, 0]
    swimming_time = data[:, 3]
    cycling_time = data[:, 5]
    running_time = data[:, 7]

    # Calculate covariance and correlation
    cov_total_rank_total_time = covariance(total_rank, total_time)
    corr_total_rank_total_time = correlation(total_rank, total_time)

    cov_age_total_time = covariance(age, total_time)
    corr_age_total_time = correlation(age, total_time)

    cov_total_time_swimming_time = covariance(total_time, swimming_time)
    corr_total_time_swimming_time = correlation(total_time, swimming_time)

    cov_cycling_time_running_time = covariance(cycling_time, running_time)
    corr_cycling_time_running_time = correlation(cycling_time, running_time)

    # Convert total time to minutes and redo the calculations
    total_time_minutes = total_time / 60
    cov_age_total_time_minutes = covariance(age, total_time_minutes)
    corr_age_total_time_minutes = correlation(age, total_time_minutes)

    # Print results
    print(f"The covariance between total rank and total time is {cov_total_rank_total_time:.2f}.")
    print(f"The correlation between total rank and total time is {corr_total_rank_total_time:.2f}.")
    print("\n")

    print(f"The covariance between age and total time is {cov_age_total_time:.2f}.")
    print(f"The correlation between age and total time is {corr_age_total_time:.2f}.")
    print("\n")

    print(f"The covariance between total time and swimming time is {cov_total_time_swimming_time:.2f}.")
    print(f"The correlation between total time and swimming time is {corr_total_time_swimming_time:.2f}.")
    print("\n")

    print(f"The covariance between cycling time and running time is {cov_cycling_time_running_time:.2f}.")
    print(f"The correlation between cycling time and running time is {corr_cycling_time_running_time:.2f}.")
    print("\n")

    print(f"The covariance between age and total time (in minutes) is {cov_age_total_time_minutes:.2f}.")
    print(f"The correlation between age and total time (in minutes) is {corr_age_total_time_minutes:.2f}.")
    print("\n")

    # e)


def ex2():
    radiation = np.loadtxt("radiation.txt")


if __name__ == '__main__':
    ex1()
    #ex2()
