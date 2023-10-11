import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math


class Gaussian:
    """
    Multi-variable (n) gaussian distribution.
    :attribute n: number of variable
    :attribute mean:
    :attribute cov: covariance
    """
    def __init__(self, n, mean, cov):
        self.n = n
        self.mean = np.array(mean)
        self.cov = np.array(cov)

    def pdf(self, sample):
        """
        compute p(sample|mean, cov).
        :param sample: multivariable sample from Gaussian
        :return: probability density function value of the sample.
        """
        sample = np.array(sample)
        return 1 / (math.pow((2*np.pi), self.n / 2) * np.sqrt(np.linalg.det(self.cov))
                    ) * np.exp(-0.5 * (sample - self.mean).dot(np.linalg.inv(self.cov)).dot((sample - self.mean).T))

    def loglikelihood(self, prior, sample):
        """

        :param prior: prior probability.
        :param sample: sample to compute logarithm likelihood.
        :return: compute result.
        """
        return -1 / 2 * (sample - self.mean) @ np.linalg.inv(self.cov) @ (sample - self.mean).T - self.n / 2 * \
            np.log(2*np.pi) - 1 / 2 * np.log(np.linalg.det(self.cov)) + np.log(prior)

    def plot_pdf(self):
        """
        plot the pdf of the gaussian.
        :return: none
        """
        x_coordinate = np.linspace(-5, 5, 100).reshape(100, 1)
        y_coordinate = np.linspace(-5, 5, 100).reshape(100, 1)
        x_coordinate, y_coordinate = np.meshgrid(x_coordinate, y_coordinate)
        likelihood = np.zeros(shape=(100, 100))

        # how to optimize? vectorize
        for idx1 in range(x_coordinate.shape[0]):
            for idx2 in range(x_coordinate.shape[1]):
                likelihood[idx1, idx2] = self.pdf([x_coordinate[idx1, idx2], y_coordinate[idx1, idx2]])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 8))
        ax.plot_surface(x_coordinate, y_coordinate, likelihood, vmin=likelihood.min() * 2, cmap=cm.Blues)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Probability Density Function of Gaussian at 0x%s' % (id(self)))

        ax.set(xticklabels=[],
               yticklabels=[],
               zticklabels=[])

        plt.show()

    def plot_identical_pdf(self):
        """
        plot the contour line of the gaussian.
        :return: none
        """
        x_coordinate = np.linspace(-5, 5, 100).reshape(100, 1)
        y_coordinate = np.linspace(-5, 5, 100).reshape(100, 1)
        x_coordinate, y_coordinate = np.meshgrid(x_coordinate, y_coordinate)
        likelihood = np.zeros(shape=(100, 100))

        # how to optimize? vectorize
        for idx1 in range(x_coordinate.shape[0]):
            for idx2 in range(x_coordinate.shape[1]):
                likelihood[idx1, idx2] = self.pdf([x_coordinate[idx1, idx2], y_coordinate[idx1, idx2]])

        fig, ax = plt.subplots(figsize=(8, 8))
        levels = np.linspace(likelihood.min(), likelihood.max(), 10)

        ax.contourf(x_coordinate, y_coordinate, likelihood, levels=levels)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('contour of Gaussian at 0x%s' % (id(self)))
        plt.show()
