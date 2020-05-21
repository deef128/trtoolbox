import sys
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd
from PyQt5 import QtCore, QtGui, QtWidgets, uic

qtCreatorFile = "gui/interface.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class DataStorage():
    def __init__(self):
        self.data = np.array([])
        self.time = np.array([])
        self.wn = np.array([])

class Results():
    def __init__(self):
        self.svd = mysvd.Results()

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.actionLoad_data.triggered.connect(self.get_files)
        self.doSVD.clicked.connect(self.svd)
        self.data = DataStorage()
        self.results = Results()
        
    def get_files(self):
        # fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open data file', "ASCII files (*.dat *.txt)")
        # self.data.data = np.loadtxt(fname[0], delimiter=',')
        # fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open time file', "ASCII files (*.dat *.txt)")
        # self.data.time = np.loadtxt(fname[0], delimiter=',')
        # fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open frequency file', "ASCII files (*.dat *.txt)")
        # self.data.wn = np.loadtxt(fname[0], delimiter=',')
        self.data.data = np.loadtxt('./data/data.dat', delimiter=',')
        self.data.time = np.loadtxt('./data/time.dat', delimiter=',')
        self.data.wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')

    def svd(self):
        print('SVD')
        # plt.figure()
        self.results.svd = mysvd.dosvd(self.data.data, self.data.time, self.data.wn, n=5)
        self.results.svd.plot_results()
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_()) 
