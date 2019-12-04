import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def func(x):
    return np.sin(x)**2 + np.cos(x)**3

def polynomial(x, coeffs):
    y = np.zeros(x.size)
    expos = np.arange(len(coeffs))
    for i, coeff in enumerate(coeffs):
        y += x**expos[i]*coeff
    return y

def integrate(dt, n):
    intervals = np.linspace(0, dt, n+1)
    xs = (intervals[:-1] + intervals[1:]) / 2
    return sum(func(xs)) * dt/n

def fit_and_plot_polynomial(x, y, poly_deg):

    plot_xs = np.linspace(min(x), max(x), 1000)

    fit_coeffs = np.polyfit(x, y, poly_deg)
    fit_coeffs = np.flip(fit_coeffs)

    plt.plot(plot_xs, polynomial(plot_xs, fit_coeffs))
    plt.plot(x, y, 'ob')
    plt.show()

ks = np.arange(1, 6)
ns = 2**ks
xs = 1/ns
xs = np.append(np.array([1]), xs)
print(xs)
ys = np.zeros(xs.size)

for i, n in enumerate(ns):
    ys[i] = integrate(1, n)

fit_and_plot_polynomial(xs, ys, 2)