import trtoolbox.mysvd as mysvd
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('./data/data.dat', delimiter=',')
wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')
time = np.loadtxt('./data/time.dat', delimiter=',')

# mysvd.show_svs(data, time, wn)
res = mysvd.reconstruct(data, 5)
res.wn = wn
res.time = time
# res.plotdata()
res.plotsvddata()
plt.show()

# res = mysvd.dosvd(data, time, wn, [1, 2, 3, 5])
# plt.show()
