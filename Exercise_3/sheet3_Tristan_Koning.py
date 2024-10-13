"""Skeletоn sheet 3 Datenanalyse University of Zurich"""
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def integrate(dist, lower, upper):
    """Integrate the pdf of a distribution between lower and upper.

    Parameters
    ----------
    dist : scipy.stats.rv_continuous
        A scipy.stats distribution object.
    lower : float
        Lower limit of the integration.
    upper : float
        Upper limit of the integration.

    Returns
    -------
    integral : float
        The integral of the pdf between lower and upper.
    """
    num_bins = 1000
    x = np.linspace(lower, upper, num_bins)
    y = dist.pdf(x)
    dx = (upper - lower) / (num_bins - 1)
    integral = np.trapz(y, dx=dx)
    return integral


# THIS FUNCTION IS NOT NEEDED, JUST DEMONSTRATION PURPOSE
def example_integrate_shifted_norm(x):
    # to get a "norm distribution" with mean 5 and std 3, we can use
    norm_dist_shifted = scipy.stats.norm(loc=5, scale=3)
    # we can then use different methods of the norm_dist_shifted object to calculate
    # the probability density function (pdf) and the cumulative distribution function (cdf)
    # and more.
    # using the cdf we can also calculate the integral of the pdf:
    integrate_4to7 = norm_dist_shifted.cdf(7) - norm_dist_shifted.cdf(4)  # integral form 4 to 7
    # or just write a function that does it for us
    integral_1to10 = integrate(norm_dist_shifted, 1, 10)
    integral_4to7 = integrate(norm_dist_shifted, 4, 7)

    print(integrate_4to7)
    print(integral_4to7)


def plot_binomial(probability: float, trials: int, plot_name: str):
    """Plot the probability mass function of a binomial distribution.
    :param probability: The probability of success.
    :param trials: The number of trials."""
    # Setup
    dist = scipy.stats.binom(n=trials, p=probability)
    x = np.arange(0, trials + 1)
    y = dist.pmf(x)

    # Plot the distribution
    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Number of Detector Hits")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution of n={trials}, p={probability}")
    plt.savefig(f"{plot_name}.png")
    plt.close()


def get_min_detectors(probability: float, required_hits: int, target_efficiency: float) -> int:
    """Calculate the minimum number of detectors required to achieve a target efficiency.
    :param probability: The probability of a detector hit.
    :param required_hits: The number of required detector hits.
    :param target_efficiency: The target efficiency.
    :return: The minimum number of detectors required."""
    num_detectors = required_hits
    while True:
        # Check if at least required_hits are detected with target_efficiency
        dist = scipy.stats.binom(n=num_detectors, p=probability)
        efficiency = 1 - dist.cdf(required_hits - 1)
        if efficiency >= target_efficiency:
            break
        num_detectors += 1

    return num_detectors


def plot_detected_particles(nr_particles: int, nr_detectors: int, probability: float, required_hits: int,
                            plot_name: str):
    """Plot the probability of detecting a particle with a certain number of detectors.
    :param nr_particles: The number of particles to detect.
    :param nr_detectors: The number of detectors.
    :param probability: The probability of a detector hit.
    :param required_hits: The number of required detector hits."""
    # Setup
    dist = scipy.stats.binom(n=nr_detectors, p=probability)

    # Calculate the probability of detecting one particle with a certain number of detectors and minimum required htis
    probability_detect_particle = 1 - dist.cdf(required_hits - 1)

    # Calculate the probability of detecting a number of particles
    particle_detection_dist = scipy.stats.binom(n=nr_particles, p=probability_detect_particle)
    x = np.arange(0, nr_particles + 1)
    y = particle_detection_dist.pmf(x)

    # Plot the distribution
    plt.figure()
    plt.bar(x, y)
    plt.xlabel("Number of Detected Particles")
    plt.ylabel("Probability")
    plt.title(f"Number of Detected Particles with {nr_detectors} Detectors")
    plt.savefig(f"{plot_name}.png")
    plt.close()


def detected_ZBosons(prob, N, min_detected):
    binom_distribution = scipy.stats.binom(n=N, p=prob)
    prob_a = 1 - binom_distribution.cdf(min_detected - 1)
    print(f"4a) {prob_a:.3f}% probability to have {min_detected} or more Z-Bosons detected")

    return binom_distribution


def gaussian_approx_ZBosons(prob, N, min_detected):
    expected_value = prob * N
    std = sqrt(N * prob * (1 - prob))
    gaussian_distribution = scipy.stats.norm(loc=expected_value, scale=std)
    prob_b = integrate(gaussian_distribution, min_detected, N)
    print(f"4b) {prob_b:.3f}% probability to have {min_detected} or more Z-Bosons detected")

    return gaussian_distribution


def plot_binom_gaussian_approximation(binom_distribution, prob, N, min_detected, plot_name):
    # Gaussian Approximation
    expected_value = prob * N
    std = sqrt(N * prob * (1 - prob))
    gaussian_distribution = scipy.stats.norm(loc=expected_value, scale=std)
    prob_b = integrate(gaussian_distribution, min_detected, N)
    print(f"\n4b) {prob_b:.3f}% probability to have {min_detected} or more Z-Bosons detected")

    # Setup Plot
    x = np.arange(0, N + 1)
    y_binom = binom_distribution.pmf(x)
    y_gaussian = gaussian_distribution.pdf(x)

    # Plot the distributions
    plt.figure()
    # Plot the binomial as bar plot, gaussian as a density line
    plt.bar(x, y_binom, label="Binomial Distribution", color="blue")
    plt.plot(x, y_gaussian, label="Gaussian Distribution", color="red")
    plt.xlabel("Number of Detector Hits")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution vs. Gaussian Approximation of n={N}, p={prob}")
    plt.legend()
    plt.savefig(f"{plot_name}.png")
    plt.close()


def plot_binom_poisson_approximation(binom_distribution, prob, N, min_detected, plot_name):
    # Poisson
    mean = prob * N
    poisson_distribution = scipy.stats.poisson(mean)
    prob_c = 1 - poisson_distribution.cdf(min_detected - 1)
    print(f"\n4c) {prob_c:.3f}% probability to have {min_detected} or more Z-Bosons detected")

    # Setup Plot
    x = np.arange(0, N + 1)
    y_binom = binom_distribution.pmf(x)
    y_poisson = poisson_distribution.pmf(x)

    # Plot the distributions
    plt.figure()
    # Plot the binomial as bar plot, gaussian as a density line
    plt.bar(x, y_binom, label="Binomial Distribution", color="blue")
    plt.plot(x, y_poisson, label="Poisson Distribution", color="red")
    plt.xlabel("Number of Detector Hits")
    plt.ylabel("Probability")
    plt.title(f"Binomial Distribution vs. Poisson Approximation of n={N}, p={prob}")
    plt.legend()
    plt.savefig(f"{plot_name}.png")
    plt.close()


def plot_binom_poisson_decayed(prob, N, min_decayed, plot_name):
    time = 125
    produced_bosons = 500
    mean = 500 / 125  # Bosons / hour

    binom_distribution = scipy.stats.binom(n=mean, p=prob)
    poisson_distribution = scipy.stats.poisson(mean * prob)
    prob_binomial = 1 - binom_distribution.cdf(min_decayed - 1)
    prob_poisson = 1 - poisson_distribution.cdf(min_decayed - 1)

    print(
        f"\n4d) {prob_binomial:.3f}% probability to have {min_decayed} or more Z-Bosons decayed into Neutrinos in first hour "
        f"of experiment using Binomial")
    print(
        f"4d) {prob_poisson:.3f}% probability to have {min_decayed} or more Z-Bosons decayed into Neutrinos in first hour "
        f"of experiment using Poisson")

    # Setup Plot
    x = np.arange(0, N + 1)
    y_binom = binom_distribution.pmf(x)
    y_poisson = poisson_distribution.pmf(x)

    # Plot the distributions
    plt.figure()
    # Plot the binomial as bar plot, gaussian as a density line
    plt.bar(x, y_binom, label="Binomial Distribution", color="blue")
    plt.plot(x, y_poisson, label="Poisson Distribution", color="red")
    plt.xlabel("Number of Detector Hits")
    plt.ylabel("Probability")
    plt.title(f"Decay of Bosons into Neutrinos, Binomial Distribution vs. Poisson Approximation of n={mean}, p={prob}")
    plt.legend()
    plt.savefig(f"{plot_name}.png")
    plt.close()


def ex1():
    print("Exercise 1\n")
    # (a)
    probability = 0.85
    trials = 4
    plot_name = "binomial_n4_p85"
    plot_binomial(probability, trials, plot_name)
    print(f"1a) Result can be found in {plot_name}.png\n")

    # (b)
    required_hits = 3
    target_efficiency = 0.99
    num_detectors = get_min_detectors(probability, required_hits, target_efficiency)
    print(f"1b) The minimum number of detectors required is {num_detectors}\n")

    # (c)
    nr_particles = 1000
    nr_detectors = 4
    required_hits = 3
    plot_name = "detected_particles"
    plot_detected_particles(nr_particles, nr_detectors, probability, required_hits, plot_name)
    print(f"1c) The standard deviation of the binomial distribution is {sqrt(nr_particles * probability * (1 - probability))}")
    print(f"The standard deviation of the poisson distribution is {sqrt(nr_particles * probability)}")
    print("The standard deviations do not match closely, therefore the width of the distribution is not what one would "
          "expect from the poisson distribution\n")
    print("Exercise 1 done\n")


def ex3():
    print("Exercise 3\n")
    # Setup
    mu = 1
    std = 0.01
    norm_distribution = scipy.stats.norm(loc=mu, scale=std)

    interval = [0.97, 1.03]
    prob_a = integrate(norm_distribution, interval[0], interval[1])
    print(f"3a) {prob_a:.3f}% probability to be within {interval}\n")

    interval = [0.99, 1.00]
    prob_b = integrate(norm_distribution, interval[0], interval[1])
    print(f"3b) {prob_b:.3f}% probability to be within {interval}\n")

    interval = [0.95, 1.05]
    prob_c = integrate(norm_distribution, interval[0], interval[1])
    print(f"3c) {prob_c:.3f}% probability to be within {interval}\n")

    interval = [0, 1.015]
    prob_d = integrate(norm_distribution, interval[0], interval[1])
    print(f"3d) {prob_d:.3f}% probability to have a height less than {interval[1]}\n")
    print("Exercise 3 done\n")


def ex4():
    # Setup
    print("Exercise 4\n")
    prob = 0.82
    N = 500
    min_detected = 390

    # a)
    binom_dist = detected_ZBosons(prob, N, min_detected)

    # b)
    plot_name = "Binomial Distribution vs Gaussian approximate"
    plot_binom_gaussian_approximation(binom_dist, prob, N, min_detected, plot_name)

    print(
        f"4b) Looking at the plot {plot_name}.png, we can observe that the Gaussian Approximation (red density line), "
        f"approximates the Binomial Distribution (Blue Bar Plot) quite well, with just minor deviation which can be"
        f" most easily seen at the peak (mean).")

    # c)
    plot_name = "Binomial Distribution vs Poisson approximate"
    plot_binom_poisson_approximation(binom_dist, prob, N, min_detected, plot_name)
    print(f"4c) Looking at the plot {plot_name}.png, we can observe that the Poisson approximation (Red density line) "
          f"is not similar to the original Binomial Distribution. The trend is the same, but the mean is a lot lower "
          f"and the variance a lot larger. This is due to the fact that the probability of success is quite high (0.82)."
          f" In the case where probability is very small, the poisson distribution will look more similar to the "
          f"binomial.")

    # d)
    prob = 0.18
    min_decayed = 1
    plot_name = "Decay of Bosons into Neutrinos, Binomial Distribution vs Poisson approximate"
    plot_binom_poisson_decayed(prob, N, min_decayed, plot_name)
    print(f"4d) Looking at the plot {plot_name}.png, we can observe that the poisson and binomial distributions are "
          f"very similar to eachother. The result is different to c) because we are now interested in the decay into "
          f"Neutrinos which we cannot detect, so we inverse the probability (0.18 insteadof 0.82). Since this is a very"
          f" low probability, the poisson distribution can approximate the binomial well\n")

    print("Exercise 4 done")


if __name__ == '__main__':
    ex1()
    ex3()  # uncomment to run ex3
    ex4()  # uncomment to run ex4
