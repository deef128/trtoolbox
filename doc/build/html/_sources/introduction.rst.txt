.. highlight:: rst

Introduction
============

All analysis routines can be found in ./trtoolbox/\*.py. The overall theme of this package is that the main routines can be run with *module.do\*(data, time, freq, kwargs)* and requires numpy arrays as input. The data layout in mind while writing the code is that frequency spans over rows and time over columns. It should also work if axis are swapped but it could well be that I forgot somewhere to check for that. The routines will return a corresponding results object containing results and important information as attributes. Additionally, it provides convient methods for plotting the results callable via *resobj.plot_\*()*.

Example scripts are provided for each module for a hands-on experience on how to use this toolbox. The *data_generator* module found in ./test can be used to generate some spectroscopic data which is helpful to train yourself in analyzing your data with this toolbox.

Using this toolbox requires following packages:
    - numpy (>=1.18.1)
    - scipy (>=1.4.1)
    - matplotlib (>=3.1.3)
    - pyqt (>=5.9.2; just for the GUI)
