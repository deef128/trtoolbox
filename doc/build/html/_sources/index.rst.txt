.. Time-resolved Toolbox documentation master file, created by
   sphinx-quickstart on Sat Apr 18 10:48:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Time-resolved Toolbox's documentation!
=================================================

Toolbox for time-resolved spectroscopic data written in Python containg SVD, global fitting and lifetime density analysis. It is optimized for "fast" spectroscopic techniques ranging from nano- to seconds (like step-scan/rapid-scan FTIR). This is different from other implementations which are often optimized for ultra-fast techniques (pico- to nanoseconds). A QT GUI offers an intuitive way to analyze your data with this toolbox.

This project was inspired by OPTIMUS written by Dr. Chavdar Slavov (https://optimusfit.org/). The global fit routine is based on ideas by Victor A. Lorenz Fonfria (https://orcid.org/0000-0002-8859-8347). The LDA routine takes the approach of augmented matrices as documented by the well written PyLDM package by Gabriel F. Dorlhiac (https://doi.org/10.1371/journal.pcbi.1005528).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   introduction.rst
   svd.rst
   ga.rst
   lda.rst
   gui.rst
   dgen.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
