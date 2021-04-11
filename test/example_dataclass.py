# %% import stuff
from trtoolbox.pclasses import Data
from test.data_generator import DataGenerator
import matplotlib.pyplot as plt
plt.close('all')

# %% change ipython backend
try:
    get_ipython().run_line_magic('matplotlib', 'widget')
except NameError:
    pass

# %% generate data with parallel decaying processes
dgen = DataGenerator()
dgen.gen_data(
    wnlimit=[800, 1900],
    tcs=3,
    num_peaks=5,
    diff=True,
    avg_width=200,
    avg_std=15,
    noise=True,
    noise_scale=0.175,
    style='dec'
)

rawdata = Data()
rawdata.data, rawdata.time, rawdata.wn = dgen.data, dgen.time, dgen.wn
rawdata.check_input()

rawdata.plot_rawdata()
rawdata.plot_traces()
rawdata.plot_spectra()
plt.show()