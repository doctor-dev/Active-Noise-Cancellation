import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

N = 400 # Input size
K = 31 # Filter size
x = np.random.randn(N) # Input to the filter
h = signal.firwin(K, 0.5) # FIR system to be identified
t = signal.convolve(x, h) # Target output signal
t = t + 0.01 * np.random.randn(len(t)) # with added noise
mu = 0.1 # LMS step size
fig = plt.figure()
plt.title('Unknown filter')
plt.stem(h,[i for i in range(K)])
w = np.zeros(K) # Initial filter
e = np.zeros(N-K)
for n in range(0, N-K):
    xn = x[n+K:n:-1]
    en = t[n+K] - np.dot(xn , w) # Error
    w = w + mu * en * xn # Update filter (LMS algorithm)
    e[n] = en # Record error
# Plot updated filter after each iteration
    if (n % 50 == 0):
        plt.figure()
        plt.title('Estimated filter at iteration %d' % n)
        plt.plot( [i for i in range(K)], w)
        plt.figure()
        plt.title('Error signal')
        plt.plot([i for i in range(N - K)], e)
        plt.show()