import numpy as np
import matplotlib.pyplot as plt


def ex4():
    print("\nExercise 4")
    print("\n4.a)")

    # Read sand.txt
    data = np.loadtxt('sand.txt')
    diameter = data[:, 0]
    slope = data[:, 1]
    slope_uncertainty = data[:, 2]

    # Setup linear model
    m = 16.1
    m_uncertainty = 1.0
    q = -2.61
    q_uncertainty = 0.34

    # Plot fitted line
    x = np.linspace(0, 1, 100)
    y = m * x + q
    plt.figure()
    plt.plot(x, y, label='Fitted line')
    plt.xlabel('diameter')
    plt.ylabel('slope')
    plt.errorbar(diameter, slope, yerr=slope_uncertainty, fmt='o', label='Datapoints', capsize=5, elinewidth=1)
    plt.legend()
    plt.title("Fitted line with data points")
    plt.savefig('fitted_line.png')
    plt.close()

    print("The data and the fitted line is shown in the file 'fitted_line.png'")

    print("\n4.b)")

    given_x = 1.5
    predicted_y = m * given_x + q
    predicted_y_uncertainty = np.sqrt((given_x ** 2 * m_uncertainty ** 2) + q_uncertainty ** 2)

    print(f"Predicted value for x={given_x} is {predicted_y:.3f} with uncertainty {predicted_y_uncertainty:.3f}")
    print("\nThe uncertainty is calculated by the formula: sqrt((x^2 * sigma(m)^2) + sigma(q)^2)")

    print("\n4.c)")

    covariance = np.array([[1.068, -0.302], [-0.302, 0.118]])
    predicted_y_uncertainty = np.sqrt((given_x ** 2 * covariance[0, 0] ** 2) + (covariance[1, 1] ** 2) + (2 * given_x * covariance[0, 1]))

    print(f"Predicted value for x={given_x} is {predicted_y:.3f} with uncertainty {predicted_y_uncertainty:.3f}")
    print("\nThe uncertainty is calculated by the formula: sqrt((x^2 * sigma(m)^2) + sigma(q)^2 + 2 * x "
          "* covariance(m, q)), since we now also have to consider the correlation")

    print("\n We achieve a smaller uncertainty in the prediction when considering the correlation between the "
          "parameters. This is due to the fact that the covariance between m and q is negative, therefore the "
          "uncertainty is reduced as it is being utilized in the term 2 * x * covariance(m, q), which therefore also "
          "turns negative.")


if __name__ == '__main__':
    ex4()
