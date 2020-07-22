# %% import stuff
import trtoolbox.svd as mysvd
from test.data_generator import DataGenerator
import matplotlib.pyplot as plt
plt.close('all')

# %% change ipython backend
try:
    get_ipython().run_line_magic('matplotlib', 'widget')
except NameError:
    pass

# %% generate data
dgen = DataGenerator()
dgen.gen_data(
    wnlimit=[1200, 1700],
    num_peaks=3,
    tcs=[1e-6, -1, -1, 1e-1],
    diff=True,
    avg_width=150,
    avg_std=10,
    noise=True,
    noise_scale=0.15
)

# %% plot generated data
dgen.print_tcs()
dgen.plot_sas()
dgen.plot_profile()
dgen.plot_data()

# %% show singular values
mysvd.show_svs(dgen.data, dgen.time, dgen.wn)

# %% reconstruct data with n singular values
res = mysvd.dosvd(dgen.data, dgen.time, dgen.wn, n=4)
res.plot_results()
# res.plot_abstract_traces()
# res.plot_abstract_spectra()
plt.show()

# %% save files
# res.save_to_files(path)

# NOTE:
# for inspecting the Results class in Spyder after interactive plotting
# please run the clean() method!
# res.clean()
