import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class PlotHelper():
    """ Object for interactive plotting. This class ensures that
    all the matplotlib objects are kept referenced which ensures
    proper function of the sliders.

    Attributes
    ----------
    fig_traces : matplotlib.figure.Figure
    ax_traces : matplotlib.axes.Axes
    l1_traces : matplotlib.lines.Line2D
    l2_traces: matplotlib.lines.Line2D
    axfreq : matplotlib.axes.Axes
    sfreq : matplotlib.widgets.Slider
    fig_spectra : matplotlib.figure.Figure
    ax_spectra : matplotlib.axes.Axes
    l1_spectra : matplotlib.lines.Line2D
    l2_spectra : matplotlib.lines.Line2D
    axtime : matplotlib.axes.Axes
    stime : matplotlib.widgets.Slider
    axcolor : str
    """

    def __init__(self):
        self.fig_traces = matplotlib.figure.Figure()
        self.ax_traces = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.l1_traces = matplotlib.lines.Line2D([], [])
        self.l2_traces = matplotlib.lines.Line2D([], [])
        self.axfreq = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.sfreq = matplotlib.widgets.Slider(self.axfreq, '', 0, 1)

        self.fig_spectra = matplotlib.figure.Figure()
        self.ax_spectra = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.l1_spectra = matplotlib.lines.Line2D([], [])
        self.l2_spectra = matplotlib.lines.Line2D([], [])
        self.axtime = matplotlib.axes.Axes(self.fig_traces, [0, 0, 0, 0])
        self.stime = matplotlib.widgets.Slider(self.axfreq, '', 0, 1)

        self.axcolor = 'lightgoldenrodyellow'

    def plot_heatmap(self, data, time, wn, title='data', newfig=True):
        """ Plots a nice looking heatmap.

        Parameters
        ----------
        data : np.array
            Data matrix subjected to SVD. Assuming *m x n* with m as frequency
            and n as time. But it is actually not important.
        time : np.array
            Time array.
        wn : np.array
            Frequency array.
        title : np.array
            Title of plot. Default *data*.
        newfig : boolean
            Setting to False prevents the creation of a new figure.

        Returns
        -------
        nothing
        """

        if newfig is True:
            plt.figure()
        # ensuring that time spans columns
        if data.shape[1] != time.size:
            data = np.transpose(data)

        if time.size == 0 or wn.size == 0:
            plt.pcolormesh(data, cmap='jet', shading='gouraud')
        else:
            plt.pcolormesh(
                time,
                wn,
                data,
                cmap='jet',
                shading='gouraud')
        plt.xscale('log')
        plt.title(title)
        # plt.show()

    def plot_traces(self, res):
        """ Plots interactive time traces.

        Parameters
        -------
        res : mysvd.Results
            Contains the data to be plotted.

        Returns
        -------
        nothing
        """

        if res.type == 'svd':
            title = 'Time traces\nblue: Raw, red: SVD'
        elif res.type == 'lda':
            title = 'Time traces\nblue: Raw, red: LDA'

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2)
        plt.plot([min(res.time), max(res.time)], [0, 0], '--', color='k')
        l1, = plt.plot(res.time, res.data[0, :])
        l2, = plt.plot(res.time, res.svddata[0, :])
        plt.xscale('log')
        ax.margins(x=0)

        axfreq = plt.axes([0.175, 0.05, 0.65, 0.03], facecolor=self.axcolor)
        sfreq = Slider(
            axfreq, res.wn_name,
            np.min(res.wn),
            np.max(res.wn),
            valinit=res.wn[0],
            valstep=abs(res.wn[1]-res.wn[0])
            )
        ymin = np.min(res.data[:, :])
        ymax = np.max(res.data[:, :])
        sc = 1.05
        ax.set_ylim(ymin*sc, ymax*sc)

        def update(val):
            val = sfreq.val
            ind = np.where(res.wn == val)[0][0]
            l1.set_ydata(res.data[ind, :])
            l2.set_ydata(res.svddata[ind, :])
            # ymin = min(res.data[ind, :])
            # ymax = max(res.data[ind, :])
            # sc = 1.1
            # ax.set_ylim(ymin*sc, ymax*sc)

        sfreq.on_changed(update)

        self.fig_traces = fig
        self.ax_traces = ax
        self.l1_traces = l1
        self.l2_traces = l2
        self.axfreq = axfreq
        self.sfreq = sfreq

    def plot_spectra(self, res):
        """ Plots interactive spectra.

        Parameters
        -------
        res : mysvd.Results
            Contains the data to be plotted.

        Returns
        -------
        nothing
        """

        if res.type == 'svd':
            title = 'Time traces\nblue: Raw, red: SVD'
        elif res.type == 'lda':
            title = 'Time traces\nblue: Raw, red: LDA'

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.subplots_adjust(bottom=0.2)
        plt.plot([min(res.wn), max(res.wn)], [0, 0], '--', color='k')
        l1, = plt.plot(res.wn, res.data[:, 0], 'o-', markersize=4)
        l2, = plt.plot(res.wn, res.svddata[:, 0], 'o-', markersize=4)
        ax.margins(x=0)

        axtime = plt.axes([0.175, 0.05, 0.65, 0.03], facecolor=self.axcolor)
        stime = Slider(
            axtime, res.time_name,
            np.log10(np.min(res.time)),
            np.log10(np.max(res.time)),
            valinit=np.log10(np.min(res.time[0])),
            valstep=0.01
            )
        stime.valtext.set_text('%1.2e' % (10**stime.val))
        ymin = np.min(res.data[:, :])
        ymax = np.max(res.data[:, :])
        sc = 1.05
        ax.set_ylim(ymin*sc, ymax*sc)

        def update(val):
            val = 10**(stime.val)
            ind = abs(val - res.time).argmin()
            stime.valtext.set_text('%1.2e' % (res.time[ind]))
            l1.set_ydata(res.data[:, ind])
            l2.set_ydata(res.svddata[:, ind])
            # ymin = min(res.data[:, ind])
            # ymax = max(res.data[:, ind])
            # sc = 1.1
            # ax.set_ylim(ymin*sc, ymax*sc)

        stime.on_changed(update)

        self.fig_spectra = fig
        self.ax_spectra = ax
        self.l1_spectra = l1
        self.l2_spectra = l2
        self.axtime = axtime
        self.stime = stime
