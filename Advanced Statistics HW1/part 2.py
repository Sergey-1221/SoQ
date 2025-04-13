import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import gamma

# Read data
x = np.loadtxt('part 2. input.csv')

# Compute constants
n = len(x)
x_bar = np.mean(x)
s_log_x = np.sum(np.log(x))

# Define negative log-likelihood function
def nll(k):
    if k <= 0:
        return np.inf  # Return infinity if k is not positive
    term1 = n * (k * (np.log(x_bar) - np.log(k) + 1) + gammaln(k))
    term2 = - (k - 1) * s_log_x
    return term1 + term2

# Optimize nll(k)
result = minimize(nll, x0=1.0, bounds=[(1e-6, None)], method='SLSQP')

# Extract estimated k
estimated_k = result.x[0]

# Compute estimated θ
estimated_theta = x_bar / estimated_k

# Print estimated values
print(estimated_k, estimated_theta)

# Visualization
# Plot histogram of data
plt.hist(x, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')

# Generate values for x-axis
x_values = np.linspace(min(x), max(x), 1000)

# Compute estimated Gamma
pdf_values = gamma.pdf(x_values, a=estimated_k, scale=estimated_theta)

# Plot estimated Gamma
plt.plot(x_values, pdf_values, 'r-', label='Estimated Gamma')

# Labels and title
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Histogram of Data with Estimated Gamma')
plt.legend()

# Show plot
plt.show()
