import numpy as np

from LDA import LDA
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron


def split_dataset(feature, label):
    data = np.hstack((feature, label))
    np.random.shuffle(data)
    train_set = data[:100, :]
    test_set = data[100:, :4]
    result = data[100:, 4:].flatten()
    return train_set, test_set, result


def standardize(feature):
    mean = np.mean(feature, axis=0)
    stddev = np.std(feature)
    processed = (feature - mean) / stddev
    return processed


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # split dataset
    train, test, res = split_dataset(X, y.reshape(-1, 1))

    # self-implemented LDA class
    feature = train[:, :4].T
    label = train[:, 4].astype(np.uint64)

    # construct model -- LDA for dimension reduction & perception for classification
    lda = LDA(2, feature, label)
    clf = Perceptron(tol=1e-3, random_state=0)

    # fitting
    lda.fit()

    # result of dimension reduction
    X_r = lda.transform(lda.data.T)
    test = lda.transform(test)

    # fitting the clf model
    clf.fit(X_r, lda.labels)

    # predict result
    result = clf.predict(test)

    # accuracy
    train_accuracy = clf.score(X_r, lda.labels)
    test_accuracy = np.sum(res == result) / 50
    print("train accuracy is {} \n test accuracy is {}".format(train_accuracy, test_accuracy))

    # plot the result of dimension reduction
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    for color, i, target_name in zip(colors, lda.all_labels, target_names):
        plt.scatter(
            X_r[label == i, 0], X_r[label == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("customized LDA of iris dataset")
    plt.show()
