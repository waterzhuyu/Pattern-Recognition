import matplotlib.pyplot as plt
import numpy as np
from gaussian import Gaussian


def gen_dataset(num_samples1: int, num_samples2: int, gaussian1: Gaussian, gaussian2: Gaussian) -> np.ndarray:
    """
    Generate two dataset corresponding to two multi-variable Gaussian distribution, then concatenate them.
    :param num_samples1: number of first dataset
    :param num_samples2:
    :param gaussian1
    :param gaussian2
    :return:
    """
    dataset1 = np.random.multivariate_normal(mean=gaussian1.mean, cov=gaussian1.cov, size=num_samples1)
    labels1 = np.zeros(num_samples1).reshape(-1, 1)
    dataset1 = np.hstack((dataset1, labels1))

    dataset2 = np.random.multivariate_normal(mean=gaussian2.mean, cov=gaussian2.cov, size=num_samples2)
    labels2 = np.ones(num_samples2).reshape(-1, 1)
    dataset2 = np.hstack((dataset2, labels2))
    return np.concatenate((dataset1, dataset2), axis=0)


def split_dataset(data_set: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Split dataset to train and test.
    :param data_set: original dataset.
    :return: train dataset and test dataset.
    """
    np.random.shuffle(data_set)
    train_set = data_set[:4000]
    test_set = data_set[4000:, :2]
    result = data_set[4000:, 2:].flatten()
    return train_set, test_set, result


def fit(data_set):
    """
    Fit the multi-variable gaussian model.
    :param data_set: input data.
    :return: gaussian model.
    """
    n = data_set.shape[1] - 1
    data1 = np.array([data for data in data_set if not data[n]])[:, :n]
    data2 = np.array([data for data in data_set if data[n]])[:, :n]

    gaussian1 = Gaussian(n, np.mean(data1, axis=0), np.cov(data1.T))
    gaussian2 = Gaussian(n, np.mean(data2, axis=0), np.cov(data2.T))

    return gaussian1, gaussian2


def bayesian_decision(prior, data: np.ndarray, gaussian1, gaussian2) -> np.ndarray:
    """
    decide the label of the data.
    :param prior: prior probability
    :param data: data set
    :param gaussian1: gaussian model 1
    :param gaussian2: gaussian model 2
    :return: decision
    """
    loglikelihood1 = np.array([gaussian1.loglikelihood(prior, sample) for sample in data])

    loglikelihood2 = np.array([gaussian2.loglikelihood(1 - prior, sample) for sample in data])
    return np.array(loglikelihood1 < loglikelihood2)


def plot_decision_hyperplane(prior, gaussian1: Gaussian, gaussian2: Gaussian):
    """
    plot the decision plane
    :param prior: prior probability
    :param gaussian1: gaussian model 1
    :param gaussian2: gaussian model 2
    :return: none
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    pos = np.dstack((x, y)).reshape((-1, 2))
    pdf1 = np.array([gaussian1.loglikelihood(prior, sample) for sample in pos])
    pdf2 = np.array([gaussian2.loglikelihood(1 - prior, sample) for sample in pos])

    decide = (pdf1 > pdf2).reshape((200, 200))
    plt.contourf(x, y, decide, levels=[-0.5, 0.5, 1.5], alpha=0.5)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Bayesian Decision Plane')
    plt.show()


if __name__ == '__main__':
    # case #1
    # gen_gaussian1 = Gaussian(2, [0, 2], [[1, 0], [0, 1]])
    # gen_gaussian2 = Gaussian(2, [5, 0], [[1, 0], [0, 1]])

    # case #2
    # gen_gaussian1 = Gaussian(2, [0, 2], [[2, 1], [1, 2]])
    # gen_gaussian2 = Gaussian(2, [5, 0], [[2, 1], [1, 2]])

    # case #3
    gen_gaussian1 = Gaussian(2, [0, 2], [[2, 1], [1, 2]])
    gen_gaussian2 = Gaussian(2, [3, 0], [[3, 2], [2, 3]])

    dataset = gen_dataset(3000, 3000, gen_gaussian1, gen_gaussian2)
    train, test, true_label = split_dataset(dataset)

    estimate_gaussian1, estimate_gaussian2 = fit(dataset)

    estimate_gaussian1.plot_pdf()
    estimate_gaussian2.plot_pdf()

    estimate_gaussian1.plot_identical_pdf()
    estimate_gaussian2.plot_identical_pdf()

    prior_prob = 1 - np.sum(true_label) / 2000  # compute the prior probability

    decision = bayesian_decision(prior_prob, test, estimate_gaussian1, estimate_gaussian2)

    true_decision = decision == true_label

    accuracy = true_decision.sum() / 2000

    plot_decision_hyperplane(prior_prob, estimate_gaussian1, estimate_gaussian2)

    print("prior is", prior_prob)
    print("accuracy is", accuracy)
