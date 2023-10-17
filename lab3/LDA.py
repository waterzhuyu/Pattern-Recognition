import numpy as np


class LDA:
    """
    Linear Discriminant Analysis.

    Attributes
    ----------
    n_components: the kept dimension of dimension reduction
    data : feature dataset
    labels : true label of dataset
    all_labels : all possible value of label
    label_nums : the number of sort of label
    projection_direction : model result, to project on this vector

    """
    def __init__(self, n_components, data, labels):
        self.n_components = n_components
        self.data = data
        self.labels = labels
        self.all_labels = np.unique(self.labels)
        self.label_nums = [np.sum(self.labels == label) for label in self.all_labels]
        self.projection_direction = 0

    def mean_of_class(self, label) -> np.ndarray:
        """
        compute the mean of label class.
        :param label: label of class
        :return: mean
        """
        assert label in self.labels, "Can't recognize label."
        return np.mean(self.data[:, self.labels == label], axis=1, keepdims=True)

    def between_class_scatter(self) -> np.ndarray:
        """
        compute the between class scatter of the dataset.
        :return: between-class scatter
        """
        total_mean = np.mean(self.data, axis=1, keepdims=True)

        scatter = np.zeros((self.data.shape[0], self.data.shape[0]))
        for label in self.all_labels:
            mean_of_label = self.mean_of_class(label)
            scatter += self.label_nums[label] * (mean_of_label - total_mean) @ (
                    mean_of_label - total_mean).T

        return scatter

    def within_class_scatter(self) -> np.ndarray:
        """
        compute the within class scatter of the class
        :return: within-class scatter
        """
        scatter = np.zeros((self.data.shape[0], self.data.shape[0]))
        for label in self.all_labels:
            # mean_of_label = self.mean_of_class(label)
            # for x in self.data[:, label == self.labels].T:
            #     scatter += (mean_of_label - x) @ (mean_of_label - x).T
            class_scatter = np.cov(self.data.T[self.labels == label], rowvar=False)
            scatter += class_scatter

        return scatter

    def fit(self):
        """
        fit the linear discriminant analysis model
        :return: none
        """
        lambdas, eigen_vectors = np.linalg.eigh(np.linalg.inv(self.within_class_scatter()) @ self.between_class_scatter())
        # print(lambdas)
        # print(eigen_vectors)
        idx = np.argsort(lambdas)
        # print(idx)
        eigen_vectors = eigen_vectors[:, np.argsort(lambdas)]
        # print(eigen_vectors)
        self.projection_direction = eigen_vectors[-self.n_components:, :]

    def transform(self, data):
        # self.data = self.data.T @ self.projection_direction.T
        return data @ self.projection_direction.T
