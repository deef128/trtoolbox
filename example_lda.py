import trtoolbox.mylda as mylda
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

res = mylda.dolda(data, time, wn, tlimits=[1e-6, 1e-1])
print(res.x_k.shape)
res.plotlda()
plt.show()
