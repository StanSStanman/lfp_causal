import matplotlib.pyplot as plt
import numpy as np

X = np.arange(0.1, 0.9, 0.1)
Y = np.arange(0.1, 0.9, 0.1)

# dP
dP = np.zeros([len(X), len(Y)])
for i in np.arange(len(X)):
    for j in np.arange(len(Y)):
        dP[i, j] = X[i] - Y[j]

# log dP
logdP = np.zeros([len(X), len(Y)])
k = 0
for i in np.arange(len(X)):
    for j in np.arange(len(Y)):
        logdP[i, j] = np.log(X[i]) - np.log(Y[j])

# Instrumental Jensen-Shannon Divergence
IS = np.zeros([len(X), len(Y)])
for i in np.arange(len(X)):
    for j in np.arange(len(Y)):
        Ps1 = 0.5 * (X[i] + Y[j])
        Ps2 = 0.5 * ((1.-X[i]) + (1.-Y[j]))
        ID1 = 0.5 * np.sum([ np.log(X[i]/Ps1) * X[i], np.log((1.-X[i])/Ps2) * (1.-X[i]) ])
        ID2 = 0.5 * np.sum([ np.log(Y[i]/Ps1) * Y[i], np.log((1.-Y[i])/Ps2) * (1.-Y[i]) ])
        IS[i, j] = ID1 + ID2

# Plotting 1
plt.subplot(2, 2, 1)
plt.plot(dP, IS, '.b')
plt.xlabel('dP')
plt.ylabel('IS')
plt.grid(True)

# Plotting 2
plt.subplot(2, 2, 2)
plt.plot(logdP, IS, '.b')
plt.xlabel('Log dP')
plt.ylabel('IS')
plt.grid(True)

# Plotting 3
plt.subplot(2, 2, 3)
plt.plot(np.abs(logdP), IS, '.b')
plt.xlabel('Abs Log dP')
plt.ylabel('IS')
plt.grid(True)

plt.show()