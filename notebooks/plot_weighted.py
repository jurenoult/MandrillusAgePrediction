from matplotlib import pyplot as plt


actual_max_age = 4*365
max_age = 4*365
min_age = 0
gamma = 2

def get_weight_function(gamma, y_max):
    def weight(y):
        return max(1e-4 , (1.0 - (y/y_max))**gamma)
    return weight

def get_mse(y, y_true):
    diff = y - y_true
    return diff ** 2

def get_weighted_mse_function(weight_function):
    def wmse(y, y_true):
        weight = weight_function(y_true)
        return weight * get_mse(y, y_true)
    return wmse

weight_function = get_weight_function(gamma=gamma, y_max=max_age)
weighted_mse_function = get_weighted_mse_function(weight_function)

x = list(range(min_age, actual_max_age))
y = [weight_function(xi) for xi in x]

errors = [10/365, 20/365, 50/365, 100/365, 200/365]
y_errors_weighted = [ [weighted_mse_function(xi-error, xi) for xi in x] for error in errors ]
y_errors_mse = [ [get_mse(xi-error, xi) for xi in x] for error in errors ]

fig, axes = plt.subplots(1, 3, figsize=(16, 10))

axes[0].plot(x, y)
[axes[1].plot(x, y_error_weighted) for y_error_weighted in y_errors_weighted]
[axes[2].plot(x, y_error_mse) for y_error_mse in y_errors_mse]
plt.show()