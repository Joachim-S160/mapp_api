
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot():
    #  load csv files
    dataQI = np.loadtxt('output.csv', delimiter=',',
                        usecols=(2, 3), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1, 2))

    # plot errorbars
    plt.errorbar(dataQI[:, 0], dataexp[:, 0], xerr=dataQI[:, 1],
                 yerr=dataexp[:, 1], fmt='o', label='data')
    # plt.plot(dataQI[:,0], dataexp[:,0], 'ro', label='QI')
    plt.xlabel('Hong model melting point [K]')
    plt.ylabel('Experimental melting point [K]')
    plt.plot([0, 2000], [0, 2000], 'k-', label='ideal case')
    plt.legend()

    plt.show()


plot()

# TODO: add RMSE calculation


def RMSE(y_actual, y_predicted):
    MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
    return np.sqrt(MSE)


def RMSE_halides():
    dataQI = np.loadtxt('output.csv', delimiter=',', usecols=(2), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1))
    return RMSE(dataQI, dataexp)


print(RMSE_halides())
print('DFT is usually 100K off, so this is not bad')
print('The RMSE of the paper was 160K for the testing set, so the model proves to have more difficulty with the halides')

# TODO: add Pearson correlation coefficient calculation


def Pearson_correlation_coefficient(y_actual, y_predicted):
    return np.corrcoef(y_actual, y_predicted)[0, 1]


def Pearson_correlation_coefficient_halides():
    dataQI = np.loadtxt('output.csv', delimiter=',', usecols=(2), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1))
    return Pearson_correlation_coefficient(dataQI, dataexp)


print(Pearson_correlation_coefficient_halides())

# "Detailed analysis of the errors"#############################################""


def error_on_margin_(data_errors, exp_errors, y_data, predicted_y):

    # Step 1: Calculate the combined error for each data point
    combined_errors = np.sqrt(np.square(data_errors) + np.square(exp_errors))

    # Step 2: Calculate the error on the margin for each data point
    error_on_margin = (y_data - predicted_y) / combined_errors
    return error_on_margin


def error_margin_hist():

    dataQI = np.loadtxt('output.csv', delimiter=',',
                        usecols=(2, 3), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1, 2))

    y_data, data_errors = dataQI[:, 0], dataQI[:, 1]
    predicted_y, exp_errors = dataexp[:, 0], dataexp[:, 1]

    error_on_margin = error_on_margin_(
        data_errors, exp_errors, y_data, predicted_y)
    # Step 3: Compare the histogram with a standard normal distribution
    plt.hist(error_on_margin, bins=20, density=True,
             alpha=0.6, color='blue', label='Error on Margin')
    x = np.linspace(-4, 4, 100)
    # Standard normal distribution with mean 0 and standard deviation 1
    y = stats.norm.pdf(x, 0, 1)
    plt.plot(x, y, 'r', label='Standard Normal Distribution')
    plt.xlabel('Error on Margin')
    plt.ylabel('Frequency')
    plt.title('Histogram of Error on Margin vs. Standard Normal Distribution')
    plt.legend()
    plt.show()

    # This could indicate that certain measurements significantly deviate from the expected values or that there are systematic errors present in the measurements.

    # It can have various causes:

    # Outliers: There may be outliers in the data that fall far outside the normal distribution. These outliers can have a considerable impact on the histogram, leading to a block that is further away from the normal distribution.

    # Systematic errors: There might be systematic errors in the measurements, causing certain measurements to systematically deviate from the expected values. This can result in a deviation from the normal distribution in the histogram.

    # Model inaccuracies: The predictive model could be inaccurate or may not include all relevant factors, causing some measurements to deviate significantly from the expected values.

    # Small number of data points: If the number of data points is small, the histogram may appear random, and there could be parts that are further away from the normal distribution.


error_margin_hist()


def QQ_plot():

    dataQI = np.loadtxt('output.csv', delimiter=',',
                        usecols=(2, 3), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1, 2))

    y_data, data_errors = dataQI[:, 0], dataQI[:, 1]
    predicted_y, exp_errors = dataexp[:, 0], dataexp[:, 1]

    error_on_margin = error_on_margin_(
        data_errors, exp_errors, y_data, predicted_y)
    # Step 4: Create a QQ plot
    stats.probplot(error_on_margin, plot=plt)
    plt.xlabel('Theoretical Quantiles (Standard Normal Distribution)')
    plt.ylabel('Sample Quantiles (Error on Margin)')
    plt.title('QQ Plot')
    plt.show()

    # From this we see our data has bigger tails than the normal distribution, which means that our data has more outliers than the normal distribution. This could be due to the fact that we have a small dataset, and that the model is not perfect.


QQ_plot()

# TODO: in ppt explain outliers
# TiBr
# TiI4
# HfF4


def outliers():
    dataQI = np.loadtxt('output.csv', delimiter=',',
                        usecols=(2, 3), skiprows=1)
    dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1, 2))
    # load text file with molecules from csv:
    molecules = np.loadtxt('output.csv', delimiter=',',usecols=(1), skiprows=1, dtype=str)
    
    y_data, data_errors = dataQI[:, 0], dataQI[:, 1]
    predicted_y, exp_errors = dataexp[:, 0], dataexp[:, 1]
    # calculate the difference between the predicted and experimental values
    difference = y_data - predicted_y
    # calculate the standard deviation of the difference
    stdv = np.std(difference)
    indices = np.where(difference > 2*stdv)[0]
    print('The outliers are:')
    print(molecules[indices])   
outliers()

    # TODO: in ppt compare RMSE to paper
    # TODO: in ppt explain descriptors of the paper
