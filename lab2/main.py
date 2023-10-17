import numpy as np
import PdfEstimation
import pandas as pd
from matplotlib import pyplot as plt


def read_samples_from_xlsx(path):
    data = pd.read_excel(path)
    return data


if __name__ == '__main__':
    data_path = 'data/sampled_data_16.xlsx'
    # data_path = 'data/sampled_data_256.xlsx'
    # data_path = 'data/sampled_data_1000.xlsx'
    # data_path = 'data/sampled_data_2000.xlsx'
    data_sample = read_samples_from_xlsx(data_path)

    # plot the original histogram
    fis, axs = plt.subplots(2, 3, figsize=(12, 18), num='Density Estimation Result of %s' % data_path)
    plot1 = plt.subplot2grid((2, 3), (0, 0))
    plot1.hist(data_sample['Sample'])
    plot1.set_title("histogram of the original data")

    # plot the estimation result by k neighbor method
    pdf1 = PdfEstimation.PdfEstimation(data_sample.to_numpy())

    # change bandwidth here
    # bandwidth = np.sqrt(pdf1.sample.size)
    bandwidth = 3 * np.sqrt(pdf1.sample.size)
    pdf1.fit(pdf1.neighbor, bandwidth)  # k-neighbor
    plot2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
    plot2.plot(np.linspace(np.min(pdf1.sample), np.max(pdf1.sample), pdf1.sample.size), pdf1.pdf)
    plot2.set_title("estimation result using k neighbor method, k=%f" % bandwidth)

    # plot the estimation result by kernel method
    pdf2 = PdfEstimation.PdfEstimation(data_sample.to_numpy())
    for i, width in enumerate([0.25, 1, 4]):
        pdf2.fit(pdf2.kernel, [PdfEstimation.gaussian, width])  # Parzen window: kernel function = gaussian
        # pdf2.fit(pdf2.kernel, [PdfEstimation.uniform, width]) # Parzen window: kernel function = uniform
        axs[1, i].plot(np.linspace(np.min(pdf2.sample), np.max(pdf2.sample), pdf2.sample.size), pdf2.pdf)
        axs[1, i].set_title("kernel method, bandwidth=%2f" % width)

    plt.show()
