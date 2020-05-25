import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import trtoolbox.mysvd as mysvd
import trtoolbox.myglobalfit as mygf
import trtoolbox.mylda as mylda
from trtoolbox.plothelper import PlotHelper
from PyQt5 import QtWidgets, uic

Ui_MainWindow, QtBaseClass = uic.loadUiType("gui/interface.ui")


class DataStorage():
    def __init__(self):
        self.type = 'raw'
        self._data = np.array([])
        self._time = np.array([])
        self._wn = np.array([])
        self._phelper = PlotHelper()
        self.wn_name = 'wavenumber'
        self.wn_unit = 'cm^{-1}'
        self.time_name = 'time'
        self.time_unit = 's'

    @property
    def wn(self):
        return self._wn

    @wn.setter
    def wn(self, val):
        if val.dtype != 'float':
            val = val.astype('float64')
        self._wn = val.reshape((val.size, 1))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        if val.dtype != 'float':
            val = val.astype('float64')
        self._time = val.reshape((1, val.size))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if val.dtype != 'float':
            val = val.astype('float64')
        self._data = val

    def check(self):
        if self.data.size == 0 or self.time.size == 0 or self.wn.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Please load data.')
            return False
        return True

    def init_phelper(self):
        if type(self._phelper) == list:
            self._phelper = PlotHelper()

    def plot_data(self):
        self.init_phelper()
        self._phelper.plot_heatmap(
            self.data, self.time, self.wn,
            title='Raw Data', newfig=True)
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_data_3d(self):
        self.init_phelper()
        self._phelper.plot_surface(
            self.data, self.time, self.wn,
            title='Raw Data')
        plt.ylabel('%s / %s' % (self.wn_name, self.wn_unit))
        plt.xlabel('%s / %s' % (self.time_name, self.time_unit))

    def plot_spectra(self):
        """ Plots interactive spectra.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_spectra(self)

    def plot_traces(self):
        """ Plots interactive time traces.

        Returns
        -------
        nothing
        """

        self.init_phelper()
        self._phelper.plot_traces(self)


class Results():
    def __init__(self):
        self._phelper = PlotHelper()
        self.svd = mysvd.Results()
        self.gf = mygf.Results()
        self.lda = mylda.Results()

    def check_svd(self, silent=False):
        if self.svd.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do SVD.')
            return False
        return True

    def check_gf(self, silent=False):
        if self.gf.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do global fit analysis.')
            return False
        return True

    def check_lda(self, silent=False):
        if self.lda.data.size == 0:
            self.error_dialog = QtWidgets.QErrorMessage()
            if silent is False:
                self.error_dialog.showMessage('Please do LDA.')
            return False
        return True


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.data = DataStorage()
        self.results = Results()

        self.actionLoad_all.triggered.connect(self.get_file_all)
        self.actionLoad_time.triggered.connect(self.get_file_time)
        self.actionLoad_frequency.triggered.connect(self.get_file_wn)
        self.actionLoad_data.triggered.connect(self.get_file_data)
        self.actionLoad_test_data.triggered.connect(self.get_test_data)

        self.actionTime_change.triggered.connect(self.change_time)
        self.actionFreq_change.triggered.connect(self.change_freq)

        self.pb_plt_close.clicked.connect(self.close_all)
        self.pb_save_results.clicked.connect(self.save_all)

        self.pb_data_plot.clicked.connect(self.plot_raw)

        self.pb_dosvd.clicked.connect(self.dosvd)
        self.pb_singular.clicked.connect(self.show_svs)
        self.pb_svd_plot.clicked.connect(self.plot_svd)

        self.pb_dogf.clicked.connect(self.dogf)
        self.pb_gf_plot.clicked.connect(self.plot_gf)

        self.pb_lda_dolda.clicked.connect(self.dolda)
        self.pb_lda_plot.clicked.connect(self.plot_lda)

    def get_file_all(self):
        self.get_file_time()
        self.get_file_wn()
        self.get_file_data()

    def get_file_time(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open time file')
        if os.path.exists(fname[0]):
            try:
                self.data.time = np.loadtxt(fname[0], delimiter=',')
                self.label_file_time.setText(
                    'Time file:\n' + fname[0].split(os.path.sep)[-1][-20:])
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_file_wn(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open frequency file'
        )
        if os.path.exists(fname[0]):
            try:
                self.data.wn = np.loadtxt(fname[0], delimiter=',')
                self.label_file_freq.setText(
                    'Freq file:\n' + fname[0].split(os.path.sep)[-1][-20:])
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_file_data(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open data file')
        if os.path.exists(fname[0]):
            try:
                self.data.data = np.loadtxt(fname[0], delimiter=',')
                self.label_file_data.setText(
                    'Data file:\n...' + fname[0][-45:]
                )
            except ValueError:
                print('Wrong file format.')
            os.chdir(os.path.dirname(fname[0]))
        else:
            print('File not found.')

    def get_test_data(self):
        self.data.data = np.loadtxt('./data/data.dat', delimiter=',')
        self.data.time = np.loadtxt('./data/time.dat', delimiter=',')
        self.data.wn = np.loadtxt('./data/wavenumbers.dat', delimiter=',')

    def change_time(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter time unit long:'
        )
        if ok:
            self.data.time_name = text
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter time unit short:'
        )
        if ok:
            self.data.time_unit = text

    def change_freq(self):
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter frequency unit long:'
        )
        if ok:
            self.data.wn_name = text
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            'Text Input Dialog',
            'Enter frequency unit short:'
        )
        if ok:
            self.data.wn_unit = text

    def close_all(self):
        plt.close('all')

    def save_all(self):
        basepath = str(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory"
            )
        )
        if self.results.check_svd(silent=True) is True:
            path = os.path.join(basepath, 'svd')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.svd.save_to_files(path)
        if self.results.check_gf(silent=True) is True:
            path = os.path.join(basepath, 'gf')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.gf.save_to_files(path)
        if self.results.check_lda(silent=True) is True:
            try:
                alpha = float(self.txt_lda_plot_alpha.text())
            except ValueError:
                print('Please enter a valied float')
                return
            path = os.path.join(basepath, 'lda')
            if os.path.exists(path) is False:
                os.mkdir(path)
            self.results.lda.save_to_files(path, alpha=alpha)

    def plot_raw(self):
        if self.data.check() is False:
            return

        if self.cb_data_contour.isChecked() is True:
            self.data.plot_data()
        if self.cb_data_3d.isChecked() is True:
            self.data.plot_data_3d()
        if self.cb_data_spectra.isChecked() is True:
            self.data.plot_spectra()
        if self.cb_data_traces.isChecked() is True:
            self.data.plot_traces()
        plt.show()

    def show_svs(self):
        if self.data.check() is False:
            return

        mysvd.show_svs(self.data.data, self.data.time, self.data.wn)
        plt.show()

    def dosvd(self):
        if self.data.check() is False:
            return

        num = self.sb_svd_svd.value()
        self.results.svd = mysvd.dosvd(
            self.data.data,
            self.data.time,
            self.data.wn,
            n=num
        )
        self.results.svd.plot_results()
        plt.show()

    def plot_svd(self):
        if self.results.check_svd() is False:
            return

        if self.cb_svd_contour.isChecked() is True:
            self.results.svd.plot_svddata(newfig=True)
        if self.cb_svd_3d.isChecked() is True:
            self.results.svd.plot_svddata_3d()
        if self.cb_svd_spectra.isChecked() is True:
            self.results.svd.plot_spectra()
        if self.cb_svd_traces.isChecked() is True:
            self.results.svd.plot_traces()
        plt.show()

    def dogf(self):
        if self.data.check() is False:
            return

        num = self.sb_gf_svd.value()
        tcs_str = self.txt_gf_tcs.toPlainText()
        tcs_str = tcs_str.strip(' ')
        tcs_str = tcs_str.rstrip(',')
        try:
            tcs = [float(i) for i in tcs_str.split(",")]
        except ValueError:
            print('Please provide valid string')
            return
        self.results.gf = mygf.doglobalfit(
            self.data.data,
            self.data.time,
            self.data.wn,
            tcs,
            svds=num
        )
        self.results.gf.plot_results()
        plt.show()
        self.print_results()

    def print_results(self):
        str = ''
        for i, t in enumerate(self.results.gf.tcs):
            str = str + '%i. %.2e (%.2e)\n' % (i+1, t, self.results.gf.var[i])
        str = str + 'R^2 = %.2f%%' % (self.results.gf.r2)
        self.txt_gf_results.clear()
        self.txt_gf_results.insertPlainText(str)

    def plot_gf(self):
        if self.results.check_gf() is False:
            return

        if self.cb_gf_contour.isChecked() is True:
            self.results.gf.plot_fitdata()
        if self.cb_gf_3d.isChecked() is True:
            self.results.gf.plot_fitdata_3d()
        if self.cb_gf_spectra.isChecked() is True:
            self.results.gf.plot_spectra()
        if self.cb_gf_traces.isChecked() is True:
            self.results.gf.plot_traces()
        if self.cb_gf_profile.isChecked() is True:
            self.results.gf.plot_profile()
        if self.cb_gf_das.isChecked() is True:
            self.results.gf.plot_das()
        plt.show()

    def dolda(self):
        if self.data.check() is False:
            return

        tcs = []
        alpha = []

        try:
            tcs.append(float(self.txt_lda_tc_bottom.text()))
            tcs.append(float(self.txt_lda_tc_upper.text()))
            tcs_num = int(self.txt_lda_tc_number.text())

            alpha.append(float(self.txt_lda_alpha_bottom.text()))
            alpha.append(float(self.txt_lda_alpha_upper.text()))
            alpha_num = int(self.txt_lda_alpha_number.text())
        except ValueError:
            print('Just type numbers.')

        self.results.lda = mylda.dolda(
            self.data.data,
            self.data.time,
            self.data.wn,
            tlimits=tcs,
            tnum=tcs_num,
            alimits=alpha,
            anum=alpha_num
        )
        self.results.lda.plot_results()
        plt.show()

    def plot_lda(self):
        if self.results.check_lda() is False:
            return

        if self.cb_lda_results.isChecked() is True:
            self.results.lda.plot_results()

        try:
            alpha = float(self.txt_lda_plot_alpha.text())
        except ValueError:
            print('Please enter a valied float')
            return

        if self.cb_lda_contour.isChecked() is True:
            self.results.lda.plot_fitdata(alpha=alpha)
        if self.cb_lda_3d.isChecked() is True:
            self.results.lda.plot_fitdata_3d()
        if self.cb_lda_spectra.isChecked() is True:
            self.results.lda.plot_spectra(alpha=alpha)
        if self.cb_lda_traces.isChecked() is True:
            self.results.lda.plot_traces(alpha=alpha)
        if self.cb_lda_map.isChecked() is True:
            self.results.lda.plot_ldamap(alpha=alpha)
        if self.cb_lda_solution.isChecked() is True:
            self.results.lda.plot_solutionvector(alpha=alpha)
        if self.cb_lda_lcruve.isChecked() is True:
            self.results.lda.plot_lcurve()
        plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
