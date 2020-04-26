import trtoolbox.mysvd as mysvd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

mysvd.show_svs(data, time, wn)
res = mysvd.reconstruct(data, 3)
res.wn = wn
res.time = time
res.plotdata()
res.plotsvddata()
plt.show()
