# # # 

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import scipy.stats as stats

# # dataQI = np.loadtxt('output.csv', delimiter=',', usecols=(2,3), skiprows=1)
# # dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1,2))

# # y_data, data_errors = dataQI[:,0], dataQI[:,1]
# # predicted_y, exp_errors = dataexp[:,0], dataexp[:,1]

# # # Step 1: Calculate the combined error for each data point
# # combined_errors = np.sqrt(np.square(data_errors) + np.square(exp_errors))

# # # Step 2: Calculate the error on the margin for each data point
# # error_on_margin = (y_data - predicted_y) / combined_errors

# # # Step 3: Create a histogram of 'error_on_margin'
# # plt.hist(error_on_margin, bins=20, density=True, alpha=0.6, color='blue')
# # plt.xlabel('Error on Margin')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of Error on Margin')
# # plt.show()

# # # Step 4: Compare the histogram with a standard normal distribution
# # plt.hist(error_on_margin, bins=20, density=True, alpha=0.6, color='blue', label='Error on Margin')
# # x = np.linspace(-4, 4, 100)
# # y = stats.norm.pdf(x, 0, 1)  # Standard normal distribution with mean 0 and standard deviation 1
# # plt.plot(x, y, 'r', label='Standard Normal Distribution')
# # plt.xlabel('Error on Margin')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of Error on Margin vs. Standard Normal Distribution')
# # plt.legend()
# # plt.show()

# # # Step 5: Create a QQ plot
# # stats.probplot(error_on_margin, plot=plt)
# # plt.xlabel('Theoretical Quantiles (Standard Normal Distribution)')
# # plt.ylabel('Sample Quantiles (Error on Margin)')
# # plt.title('QQ Plot')
# # plt.show()



# import numpy as np

# dataQI = np.loadtxt('output.csv', delimiter=',', usecols=(2,3), skiprows=1)
# dataexp = np.loadtxt('experimental.csv', delimiter=',', usecols=(1,2))

# x_data, x_errors = dataQI[:,0], dataQI[:,1]
# y_data, y_errors = dataexp[:,0], dataexp[:,1]

# # Calculate the mean of x and y components
# mean_x = np.mean(x_data)
# mean_y = np.mean(y_data)

# # Create the covariance matrix based on the errors
# cov_matrix = np.diag(np.square(x_errors)) + np.diag(np.square(y_errors))

# # Calculate the Mahalanobis distance for each data point from the mean
# diff_matrix = np.column_stack((x_data - mean_x, y_data - mean_y))
# mahalanobis_distances = np.sqrt(np.sum(np.dot(diff_matrix, np.linalg.inv(cov_matrix)) * diff_matrix, axis=1))

# # Define a threshold for outliers (e.g., 2 or 3)
# threshold = 2

# # Find the indices of outliers based on the threshold
# outlier_indices = np.where(mahalanobis_distances > threshold)[0]

# # Get the outlier data points
# outlier_x = x_data[outlier_indices]
# outlier_y = y_data[outlier_indices]

# print("Outliers (x):", outlier_x)
# print("Outliers (y):", outlier_y)
