from math import sqrt

import numpy as np
import scipy.stats


def ex1():
    print("\nExercise 1")
    # a)
    Rk = 0.83
    sigma_rk = 0.06
    h0_rk = 1.0
    p_value = 2 * (scipy.stats.norm.cdf(Rk, h0_rk, sigma_rk))

    print(f"a)\nThe p-value is {p_value:.3f}")
    print("Since we want to test whether or not our measurement is consistent with h0=1.0, we use a two-tailed test. Our measurement is less than 1.0, therefore we can simply take 2 * cdf to get the p-value.")

    # b)
    g1 = 9.70
    sigma_g1 = 0.10
    g2 = 9.90
    sigma_g2 = 0.09

    sigma_g = sqrt(sigma_g1 ** 2 + sigma_g2 ** 2)
    difference = g1 - g2
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(difference), 0, sigma_g))

    print(f"\nb)\nThe p-value is {p_value:.3f}")
    print("Since we want to test two measurements against eachother, we have to use a t-test, since the differences could be inverted in sign but they still represent the same difference. Therefore we use a two-tailed test.")

    # c)
    h0_background = 1.5
    measured_background = 6.0
    p_value = 1 - scipy.stats.poisson.cdf(measured_background, h0_background)

    print(f"\nc)\nThe p-value is {p_value:.3f}")
    print("Since we are interested in the fact that we measured something 4 times greater than h0, we use a one-tailed test")

    # d)
    incidents_2019 = 50
    incidents_2020 = 60
    p_value = 2 * (1 - scipy.stats.poisson.cdf(incidents_2020, incidents_2019))

    print(f"\nd)\nThe p-value is {p_value:.3f}")
    print("We want to check if there is any deviation from h0, crime has stayed the same. Therefore we use a two-tailed test.")

    # e)
    infection_rate_without = 3000 / 1000000
    infection_rate_with = 3 / 8924
    p_value = 1 - scipy.stats.poisson.cdf(infection_rate_with, infection_rate_without)

    print(f"\ne)\nThe p-value is {p_value:.3f}")
    print("We are interested if the vaccine is more effective, so thta infection rates have decreased. Therefore, we have a one-tailed test.")


    # f)
    volleyball_players = [187, 185, 183, 176, 190]
    football_players = [170, 174, 186, 178, 185, 176, 182, 184, 179, 189, 177]
    N_volleyball = len(volleyball_players)
    N_football = len(football_players)

    spread_volleyball = (1 / (N_volleyball - 1)) * sum([(x - np.mean(volleyball_players)) ** 2 for x in volleyball_players])
    spread_football = (1 / (N_football - 1)) * sum([(x - np.mean(football_players)) ** 2 for x in football_players])
    S2 = ((N_volleyball - 1) * spread_volleyball + (N_football - 1) * spread_football) / (N_volleyball + N_football - 2)
    t_statistic = (np.mean(volleyball_players) - np.mean(football_players)) / (sqrt(S2) * sqrt((1 / N_volleyball) + (1 / N_football)))
    p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_statistic), N_volleyball + N_football - 2))

    print("\nf)\nCase a) Standard deviation unknown")
    print(f"The p-value is {p_value:.3f}")
    print("We want to test if the two groups have the same mean, therefore we use a two-tailed test.")

    sigma_observations = 5
    sigma = sqrt(sigma_observations ** 2 * 2)
    difference = np.mean(volleyball_players) - np.mean(football_players)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(difference), 0, sigma))

    print("\nCase b) Standard deviation known")
    print(f"The p-value is {p_value:.3f}")
    print("Since we now know the standard deviation, we can perform the test with the gaussian cdf")


def ex2():
    print("\nExercise 2")
    # a)
    counts_measured = 240
    time = 5
    microSievert_per_hour = 0.1

    # Convert to microSievert per hour
    counts_per_hour = counts_measured / time * 60
    microSievert_per_count = microSievert_per_hour / counts_per_hour
    std = sqrt(counts_measured) * microSievert_per_count

    # CI
    lower_68 = microSievert_per_hour - std
    upper_68 = microSievert_per_hour + std

    print(f"a)")
    print(f"The 68% confidence interval is [{lower_68:.5f}, {upper_68:.5f}]")

    # b)
    upper_90 = microSievert_per_hour + 1.645 * std

    print(f"\nb)")
    print(f"The 90% confidence Interval upper limit is: {upper_90:.5f}")

    # c)
    annual_limit = 1000
    hourly_limit = annual_limit / (365 * 24)

    print(f"\nc)")
    print(f"The hourly limit is: {hourly_limit:.5f}")
    print(f"The upper limit is: {upper_90:.5f}")
    print("Yes, the upper limit is below the yearly limit of 1000 microSievert")


def ex3():
    print("\nExercise 3")
    m = 90
    sigma_m = 5
    d = 5.2
    sigma_d = 0.2
    correlation = -0.6

    m_uranus = 86.8
    d_uranus = 51.1
    m_neptune = 102.0
    d_neptune = 49.5

    # Uranus hypothesis
    likelihood_uranus = scipy.stats.norm.pdf(m_uranus, m, sigma_m) + scipy.stats.norm.pdf(d_uranus, d, sigma_d)
    likelihood_neptune = scipy.stats.norm.pdf(m_neptune, m, sigma_m) + scipy.stats.norm.pdf(d_neptune, d, sigma_d)
    wilks_value = -2 * np.log(likelihood_uranus / likelihood_neptune)

    print(f"The value obtained from Wilks theorem is {wilks_value:.3f}")


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
