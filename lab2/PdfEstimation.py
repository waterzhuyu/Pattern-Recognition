import numpy as np


def get_elem(sample, idx, num):
    """
    get the element that is the nearest num element from the element at idx.
    :param sample: ndarray
    :param idx: index of operating element
    :param num: the num-nearest element for the idx element
    :return: found result
    """
    i, j = 0, 0
    cnt, offset = 1, 0
    while cnt < num:
        if idx + i - 1 >= 0 and idx + j + 1 < sample.size:
            if np.abs(sample[idx + i - 1]) < np.abs(sample[idx + j + 1]):
                i = i - 1
                offset = i
            else:
                j = j + 1
                offset = j
            cnt = cnt + 1
            continue
        if idx + i - 1 < 0:
            j = j + 1
            offset = j
        if idx + j + 1 >= sample.size:
            i = i - 1
            offset = i
        cnt = cnt + 1
    return sample[idx + offset]


def uniform(x, y: np.ndarray, band_width=0.5):
    """
    uniform kernel function.
    :param x: sample x
    :param y: all sample
    :param band_width: width of bucket
    :return: width of bucket if every dimension of x is less than y, else return 0
    """
    flag = (np.abs(x - y) < band_width / 2)
    return 1 / 2 * np.sum(flag)


def gaussian(x, y: np.ndarray, band_width=0.25):
    """
    gaussian kernel function.
    :param x: sample x
    :param y: all sample
    :param band_width: window width
    :return:
    """
    return np.sum((1 / np.sqrt(2 * np.pi)) * np.exp(-1/2 * np.square((x - y)) / band_width)) / y.size


class PdfEstimation:
    """
    Probability density function estimation non-parametric method,
    include k-neighbor and Parzen windows: uniform kernel function & gaussian kernel function.
    """
    def __init__(self, sample: np.ndarray):
        # What attributes should This class have?
        """Constructor. """
        self.sample = sample.squeeze()
        self.sample.sort()  # ndarray.sort() return none
        self.pdf = np.zeros_like(sample)

    def neighbor(self, k_n):
        # how to choose the boarder of the pdf ??? how about the multidimensional variable??
        """
        k-neighbor method, pdf[i] = \frac{k_n}{N \times V}
        :param k_n: number of sample points of a bucket
        :return: estimation result
        """
        pdf = np.zeros_like(self.sample)
        for i in range(self.sample.size):
            offset_sample = self.sample - self.sample[i]
            pdf[i] = k_n / np.abs(get_elem(offset_sample, i, k_n)) / self.sample.size

        return pdf

    def kernel(self, args):
        """
        Parzen window method.
        :param args: a list has form of [win, band_width], win is kernel function, include uniform & gaussian,
        band_width is width of band.
        :return: estimation result
        """
        assert len(args) == 2, "args should have a form of [win, band_width]"
        pdf = np.array([args[0](x, self.sample, args[1]) for x in self.sample])
        return pdf

    def fit(self, method, args):
        """
        fit the probability density function estimation model.
        :param method: k neighbor or kernel function method.
        :param args: arguments of method
        :return: None
        """
        self.pdf = method(args)
