import trtoolbox.mysvd as mysvd
import numpy as np

data = np.loadtxt('./data/data.dat', delimiter=',')
res = mysvd.reconstruct(data, 5)
print(res.svddata.shape)