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


def histogram(x, nrBins, plot_name, xLabel):
    min_x = min(x)
    max_x = max(x)

    # Calculate bin centres and height
    bin_heights = [0] * nrBins
    bin_width = (max_x - min_x) / nrBins
    bin_centres = np.arange(min_x + bin_width / 2, max_x, bin_width)


    for value in x:
        bin_index = (value - min_x) // bin_width  # Normalize to 0 as centre and divide by width of bins
        if bin_index == nrBins:
            bin_index -= 1
        bin_heights[int(bin_index)] += 1

    # Calculate errors of each bin
    bin_errors = []
    for height in bin_heights:
        bin_errors.append(np.sqrt(height))

    # Plot bars
    plt.figure()
    plt.bar(bin_centres, bin_heights, width=bin_width)
    plt.errorbar(bin_centres, bin_heights, yerr=bin_errors, fmt="o", color="r")
    plt.title(plot_name)
    plt.ylabel("Frequency")
    plt.xlabel(xLabel)
    plt.savefig(plot_name + ".pdf")
    plt.close()

    return bin_heights, bin_centres


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
    # Histogram of Age
    age_hist_heights, age_hist_bins = histogram(age, 20, "age_histogram_20_bins", "Age of Participant (years)")

    # Total time histogram
    total_time_hist_heights, total_time_hist_bins = histogram(total_time, 20, "total_time_histogram_20_bins", "Total Time (min)")

    # d)
    age_hist_mean = sum(age_hist_heights * age_hist_bins) / sum(age_hist_heights)
    age_hist_variance = sum(age_hist_heights * (age_hist_bins - age_hist_mean) ** 2) / sum(age_hist_heights)
    age_hist_std = np.sqrt(age_hist_variance)

    print(f"The mean of the age histogram with 20 bins is {age_hist_mean:.2f}, The variance is {age_hist_variance:.2f}, The standard deviation is {age_hist_std:.2f}.")

    # Test with 50 bins
    age_hist_heights, age_hist_bins = histogram(age, 50, "age_histogram_50_bins", "Age of Participant (years)")
    age_hist_mean = sum(age_hist_heights * age_hist_bins) / sum(age_hist_heights)
    age_hist_variance = sum(age_hist_heights * (age_hist_bins - age_hist_mean) ** 2) / sum(age_hist_heights)
    age_hist_std = np.sqrt(age_hist_variance)

    print(f"The mean of the age histogram with 50 bins is {age_hist_mean:.2f}, The variance is {age_hist_variance:.2f}, The standard deviation is {age_hist_std:.2f}.")

    # Test with 10 bins
    age_hist_heights, age_hist_bins = histogram(age, 10, "age_histogram_10_bins", "Age of Participant (years)")
    age_hist_mean = sum(age_hist_heights * age_hist_bins) / sum(age_hist_heights)
    age_hist_variance = sum(age_hist_heights * (age_hist_bins - age_hist_mean) ** 2) / sum(age_hist_heights)
    age_hist_std = np.sqrt(age_hist_variance)

    print(f"The mean of the age histogram with 10 bins is {age_hist_mean:.2f}, The variance is {age_hist_variance:.2f}, The standard deviation is {age_hist_std:.2f}.")
    print("\n")

    # Total Time
    total_time_hist_mean = sum(total_time_hist_heights * total_time_hist_bins) / sum(total_time_hist_heights)
    total_time_hist_variance = sum(total_time_hist_heights * (total_time_hist_bins - total_time_hist_mean) ** 2) / sum(total_time_hist_heights)
    total_time_hist_std = np.sqrt(total_time_hist_variance)

    print(f"The mean of the total time histogram with 20 bins is {total_time_hist_mean:.2f}, The variance is {total_time_hist_variance:.2f}, The standard deviation is {total_time_hist_std:.2f}.")

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


def ex2():
    radiation = np.loadtxt("radiation.txt")
    # Measurements in mSv/h
    measurements = radiation[:, 0]
    std = radiation[:, 1]

    # a)
    # Convert measurement to mSv/year
    measurements_year = measurements * 24 * 365

    # Average with uncertainties
    average_radiation = sum((1 / std**2) * measurements_year) / sum(1 / std**2)
    uncertainty_average_radiation = 1 / np.sqrt(sum(1 / std**2))

    print(f"The average radiation is {average_radiation:.6f} +/- {uncertainty_average_radiation:.6f} mSv/year.")

    # b)
    print(f"Confidence Interval of average radiation is [{(average_radiation - uncertainty_average_radiation):.6f}, {(average_radiation + uncertainty_average_radiation):.6f}].")

    # Refer to the pdf for analysis


if __name__ == '__main__':
    ex1()
    ex2()
