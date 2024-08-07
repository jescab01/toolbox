
import time
from collections import Counter
from mne import filter, time_frequency

import numpy as np
import scipy

import plotly.graph_objects as go  # for data visualisation
import plotly.io as pio
import plotly.offline
import plotly.express as px

import sys
sys.path.append("E:\\LCCN_Local\\PycharmProjects\\")
from toolbox.signals import epochingTool


## Static functional connetivity
def fc(signals, samplingFreq=None, lowcut=8, highcut=12, measure="PLV", ef=None, regionLabels=None,
       folder=None, plot=None, verbose=False, auto_open=False):
    """

    """
    tic = time.time()

    n_rois = len(signals)
    fc_matrix = np.ndarray((n_rois, n_rois))

    if "CORR" in measure:

        stdSignals = (signals - np.average(signals, axis=0)) / np.std(signals, axis=0)

        # normalSignals = np.ndarray((len(signals), len(signals[0])))
        # for channel in range(len(signals)):
        #     mean = np.mean(signals[channel, :])
        #     std = np.std(signals[channel, :])
        #     normalSignals[channel] = (signals[channel, :] - mean) / std
        print("Calculating PLV", end="") if verbose else None
        for roi1 in range(n_rois):
            print(".", end="") if verbose else None
            for roi2 in range(n_rois):
                fc_matrix[roi1][roi2] = sum(stdSignals[roi1] * stdSignals[roi2]) / len(stdSignals[0])

    elif measure in ["PLV", "AEC", "PLI"]:

        if not ef:
            filterSignals = filter.filter_data(signals, samplingFreq, lowcut, highcut, verbose=verbose)
            efSignals = epochingTool(filterSignals, 4, samplingFreq, "signals", verbose=verbose)

            # Obtain Analytical signal
            efPhase, efEnvelope = [], []
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.angle(analyticalSignal))
                efEnvelope.append(np.abs(analyticalSignal))

            # Check point
            # from toolbox import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(raw_data, raw_time, regionLabels)
            # plotConversions(raw_data[:,:len(efSignals[0][0])], efSignals[0], efPhase[0], efEnvelope[0],bands[0][b], regionLabels, 8, raw_time)
        elif "efPhase" in ef:
            efPhase = np.array(signals)[np.newaxis, :, :]
        elif "efEnvelope" in ef:
            efEnvelope = np.array(signals)[np.newaxis, :, :]

        if "PLV" in measure:
            print("Calculating PLV", end="") if verbose else None
            for roi1 in range(n_rois):
                print(".", end="") if verbose else None
                for roi2 in range(n_rois):
                    plv_values = list()
                    for epoch in range(len(efPhase)):
                        phaseDifference = efPhase[epoch][roi1] - efPhase[epoch][roi2]
                        value = abs(np.average(np.exp(1j * phaseDifference)))
                        plv_values.append(value)
                    fc_matrix[roi1, roi2] = np.average(plv_values)

        elif "AEC" in measure:
            print("Calculating AEC", end="") if verbose else None
            for roi1 in range(n_rois):
                print(".", end="") if verbose else None
                for roi2 in range(n_rois):
                    values_aec = list()  # AEC per epoch and channel x channel
                    for epoch in range(len(efEnvelope)):  # CORR between channels by epoch
                        r = np.corrcoef(efEnvelope[epoch][roi1], efEnvelope[epoch][roi2])
                        values_aec.append(r[0, 1])
                    fc_matrix[roi1, roi2] = np.average(values_aec)

        elif "PLI" in measure:
            print("Calculating PLI", end="") if verbose else None
            for roi1 in range(n_rois):
                print(".", end="") if verbose else None
                for roi2 in range(n_rois):
                    pli_values = list()
                    for epoch in range(len(efPhase)):
                        phaseDifference = efPhase[epoch][roi1] - efPhase[epoch][roi2]
                        value = np.abs(np.average(np.sign(np.sin(phaseDifference))))
                        pli_values.append(value)
                    fc_matrix[roi1, roi2] = np.average(pli_values)

    else:
        print("Unkown measure. Exit")

    if plot:
        fig = go.Figure(data=go.Heatmap(z=fc_matrix, x=regionLabels, y=regionLabels,
                                        colorbar=dict(thickness=4), colorscale='Viridis'))
        fig.update_layout(title='Phase Locking Value', height=500, width=500)

        if plot == "html":
            pio.write_html(fig, file=folder + "/" + measure + ".html", auto_open=auto_open)
        elif plot == "png":
            pio.write_image(fig, file=ffolder + "/" + measure + ".png", engine="kaleido")
        elif plot == "svg":
            pio.write_image(fig, file=folder + "/" + measure + ".svg", engine="kaleido")
        elif plot == "inline":
            plotly.offline.iplot(fig)

    print("%0.3f seconds.\n" % (time.time() - tic,)) if verbose else None

    return fc_matrix


## Dynamical functional connetivity
def dynamic_fc(data, samplingFreq, transient, window, step, measure="PLV", plot=None, folder='figures',
               lowcut=8, highcut=12, filtered=False, auto_open=False, verbose=False, mode="dFC"):
    """
    Calculates dynamical Functional Connectivity using the classical method of sliding windows.

    REFERENCE || Cabral et al. (2017) Functional connectivity dynamically evolves on multiple time-scales over a
    static structural connectome: Models and mechanisms

    :param data: Signals in shape [ROIS x time]
    :param samplingFreq: sampling frequency (Hz)
    :param window: Seconds of sliding window
    :param step: Movement step for sliding window
    :param measure: FC measure (PLV; AEC)
    :param plot: Plot dFC matrix?
    :param folder: To save figures output
    :param auto_open: on browser.
    :return: dFC matrix
    """

    window_ = window * 1000
    step_ = step * 1000

    if len(data[0]) > window_:
        if filtered:
            filterSignals = data
        else:
            # Band-pass filtering
            filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=verbose)
        if verbose:
            print("Calculating dFC matrix...")

        matrices_fc = list()
        for w in np.arange(0, (len(data[0])) - window_, step_, 'int'):

            if verbose:
                print('%s %i / %i' % (measure, w / step_, ((len(data[0])) - window_) / step_))

            efSignals = filterSignals[:, w:w + window_]

            # Obtain Analytical signal
            efPhase = list()
            efEnvelope = list()
            for i in range(len(efSignals)):
                analyticalSignal = scipy.signal.hilbert(efSignals[i])
                # Get instantaneous phase and amplitude envelope by channel
                efPhase.append(np.unwrap(np.angle(analyticalSignal)))
                efEnvelope.append(np.abs(analyticalSignal))

            # # Check point
            # from toolbox.signals import timeseriesPlot, plotConversions
            # regionLabels = conn.region_labels
            # timeseriesPlot(data, raw_time, regionLabels)
            # plotConversions(data[:, :len(efSignals[0])], efSignals, efPhase, efEnvelope, band="alpha", regionLabels=regionLabels)

            ef = "efPhase" if measure == "PLV" else "efEnvelope"
            matrices_fc.append(fc(efPhase, ef=ef, measure=measure, verbose=verbose))


        dFC_matrix = np.zeros((len(matrices_fc), len(matrices_fc)))
        for t1 in range(len(matrices_fc)):
            for t2 in range(len(matrices_fc)):
                dFC_matrix[t1, t2] = np.corrcoef(matrices_fc[t1][np.triu_indices(len(matrices_fc[0]), 1)],
                                                 matrices_fc[t2][np.triu_indices(len(matrices_fc[0]), 1)])[1, 0]

        if plot:
            fig = go.Figure(
                data=go.Heatmap(z=dFC_matrix, x=np.arange(transient, transient + len(data[0]), step_) / 1000,
                                y=np.arange(transient, transient + len(data[0]), step_) / 1000,
                                colorscale='Viridis', colorbar=dict(thickness=4)))
            fig.update_layout(title='dynamical Functional Connectivity', height=400, width=400)
            fig.update_xaxes(title="Time 1 (seconds)")
            fig.update_yaxes(title="Time 2 (seconds)")

            if plot == "html":
                pio.write_html(fig, file=folder + "/dPLV.html", auto_open=auto_open)
            elif plot == "png":
                pio.write_image(fig, file=folder + "/dPLV_" + str(time.time()) + ".png", engine="kaleido")
            elif plot == "svg":
                pio.write_image(fig, file=folder + "/dPLV.svg", engine="kaleido")
            elif plot == "inline":
                plotly.offline.iplot(fig)

        if mode == "all_matrices":
            return dFC_matrix, matrices_fc
        else:
            return dFC_matrix

    else:
        print('Error: Signal length should be longer than window length (%i sec)' % window)


def ple(efPhase, time_lag, pattern_size, samplingFreq, subsampling=1):
    """
    It calculates Phase Lag Entropy (Lee et al., 2017) on a bunch of filtered and epoched signals with shape [epoch,rois,time]
    It is based on the diversity of temporal patterns between two signals phases.

    A pattern S(t) is defined as:
        S(t)={s(t), s(t+tau), s(t+2tau),..., s(t + m*tau -1)} where
                (s(t)=1 if delta(Phi)>0) & (s(t)=0 if delta(Phi)<0)

    PLE = - sum( p_j * log(p_j) ) / log(2^m)

        where
        - p_j is the probability of the jth pattern, estimated counting the number of times each pattern
        occurs in a given epoch and
        - m is pattern size.

    REFERENCE ||  Lee et al. (2017) Diversity of FC patterns is reduced during anesthesia.


    :param efPhase: Phase component of Hilbert transform (filtered in specific band and epoched)
    :param time_lag: "tau" temporal distance between elements in pattern
    :param pattern_size: "m" number of elements to consider in each pattern
    :param samplingFreq: signal sampling frequency
    :param subsampling: If your signal has high temporal resolution, maybe gathering all possible patters is not
     efficient, thus you can omit some timepoints between gathered patterns

    :return: PLE - matrix shape (rois, rois) with PLE values for each couple; patts - patterns i.e. all the patterns
    registered in the signal and the number of times they appeared.
    """

    tic = time.time()
    try:
        efPhase[0][0][0]  # Test whether signals have been epoched
        PLE = np.ndarray((len(efPhase[0]), len(efPhase[0])))

        time_lag = int(np.trunc(time_lag * samplingFreq / 1000))  # translate time lag in timepoints
        patts = list()
        print("Calculating PLE ", end="")
        for channel1 in range(len(efPhase[0])):
            print(".", end="")
            for channel2 in range(len(efPhase[0])):
                ple_values = list()
                for epoch in range(len(efPhase)):
                    phaseDifference = efPhase[epoch][channel1] - efPhase[epoch][channel2]

                    # Binarize to create the pattern and comprehend into list
                    patterns = [str(np.where(phaseDifference[t: t + time_lag * pattern_size: time_lag] > 0, 1, 0)) for t
                                in
                                np.arange(0, len(phaseDifference) - time_lag * pattern_size, step=subsampling)]
                    patt_counts = Counter(patterns)
                    patts.append(patt_counts)
                    summation = 0
                    for key, value in patt_counts.items():
                        p = value / len(patterns)
                        summation += p * np.log10(p)

                    ple_values.append((-1 / np.log10(2 ** pattern_size)) * summation)
                PLE[channel1, channel2] = np.average(ple_values)
        print("%0.3f seconds.\n" % (time.time() - tic,))

        return PLE, patts

    except IndexError:
        print("IndexError. Signals must be epoched. Use epochingTool().")


## Metastability
def kuramoto_order(data, samplingFreq, lowcut=8, highcut=12, filtered=False, verbose=False):
    """
    From Deco et al. (2017) The dynamics of resting fluctuations in the brain: metastability and its dynamical cortical core
    "We measure the metastability as the standard deviation of the Kuramoto order parameter across time".
    """
    if verbose:
        print("Calculating Kuramoto order paramter...")
    if filtered:
        filterSignals = data

    else:
        # Band-pass filtering
        filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=verbose)

    # Padding as Hilbert transform has distortions at edges
    padding = np.zeros((len(data), 1000))

    filterSignals_padded = np.concatenate([padding, filterSignals, padding], axis=1)

    analyticalSignal_padded = scipy.signal.hilbert(filterSignals_padded)
    # Get instantaneous phase by channel
    efPhase_padded = np.angle(analyticalSignal_padded)

    efPhase = efPhase_padded[:, 1000:-1000]

    # Kuramoto order parameter in time
    kuramoto_array = abs(np.sum(np.exp(1j * efPhase), axis=0)) / len(efPhase)
    # Average Kuramoto order parameter for the set of signals
    kuramoto_avg = np.average(kuramoto_array)
    kuramoto_sd = np.std(kuramoto_array)

    return kuramoto_array, kuramoto_sd, kuramoto_avg


def kuramoto_polar(data, time_, samplingFreq, speed, lowcut=8, highcut=10, timescale="ms",
                   mode="html", folder="figures", title="", auto_open=True):
    # Band-pass filtering
    filterSignals = filter.filter_data(data, samplingFreq, lowcut, highcut, verbose=False)

    # Padding as Hilbert transform has distortions at edges
    padding = np.zeros((len(data), 1000))

    filterSignals_padded = np.concatenate([padding, filterSignals, padding], axis=1)

    analyticalSignal_padded = scipy.signal.hilbert(filterSignals_padded)
    # Get instantaneous phase by channel
    efPhase_padded = np.angle(analyticalSignal_padded)

    phases = efPhase_padded[:, 1000:-1000]

    phases = phases[:, ::speed]
    time_ = time_[::speed]

    kuramoto_order = [1 / len(phases) * np.sum(np.exp(1j * phases[:, i])) for i, dp in enumerate(phases[0])]
    KO_magnitude = np.abs(kuramoto_order)
    KO_angle = np.angle(kuramoto_order)

    wraped_phase = (phases % (2 * np.pi))
    cmap = (px.colors.qualitative.Plotly + px.colors.qualitative.Light24 + px.colors.qualitative.Set2 +
            px.colors.qualitative.Dark24 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel2 +
            px.colors.qualitative.Set3 + px.colors.qualitative.Light24 + px.colors.qualitative.Pastel1 +
            px.colors.qualitative.Alphabet + px.colors.qualitative.Vivid + px.colors.qualitative.Antique +
            px.colors.qualitative.Safe + px.colors.qualitative.D3 + px.colors.qualitative.Prism +
            px.colors.qualitative.Bold + px.colors.qualitative.G10) * 50

    # With points
    fig = go.Figure()

    ## Add Kuramoto Order
    fig.add_trace(go.Scatterpolar(theta=[KO_angle[0]], r=[KO_magnitude[0]], thetaunit="radians",
                                  name="KO", mode="markers", marker=dict(size=6, color="darkslategray")))

    ## Add each region phase
    fig.add_trace(go.Scatterpolar(theta=wraped_phase[:, 0], r=[1] * len(wraped_phase), thetaunit="radians",
                                  name="ROIs", mode="markers", marker=dict(size=8, color=cmap), opacity=0.8))

    fig.update(frames=[go.Frame(data=[go.Scatterpolar(theta=[KO_angle[i]], r=[KO_magnitude[i]]),
                                      go.Scatterpolar(theta=wraped_phase[:, i])],
                                traces=[0, 1], name=str(np.round(t, 3))) for i, t in enumerate(time_)])

    # CONTROLS : Add sliders and buttons
    fig.update_layout(template="plotly_white", height=400, width=500, polar=dict(angularaxis_thetaunit="radians", ),

                      sliders=[dict(
                          steps=[
                              dict(method='animate',
                                   args=[[str(t)], dict(mode="immediate",
                                                        frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                        transition=dict(duration=0))], label=str(np.round(t, 3))) for
                              i, t in enumerate(time_)],
                          transition=dict(duration=0), xanchor="left", x=0.35, y=-0.15,
                          currentvalue=dict(font=dict(size=15, color="black"), prefix="Time (%s) - " % (timescale),
                                            visible=True,
                                            xanchor="right"),
                          len=0.7, tickcolor="white", font=dict(color="white"))],

                      updatemenus=[
                          dict(type="buttons", showactive=False, x=0.05, y=-0.4, xanchor="left", direction="left",
                               buttons=[
                                   dict(label="\u23f5", method="animate",
                                        args=[None,
                                              dict(frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                   transition=dict(duration=0),
                                                   fromcurrent=True, mode='immediate')]),
                                   dict(label="\u23f8", method="animate",
                                        args=[[None],
                                              dict(frame=dict(duration=0, redraw=True, easing="cubic-in-out"),
                                                   transition=dict(duration=0),
                                                   mode="immediate")])])])

    if "html" in mode:
        pio.write_html(fig, file=folder + "/Animated_PolarKuramoto-" + title + ".html", auto_open=auto_open,
                       auto_play=False)

    elif "inline" in mode:
        plotly.offline.iplot(fig)


