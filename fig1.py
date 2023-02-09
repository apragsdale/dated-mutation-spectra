"""
Panels:

- Three mutation spectrum histories.
- Model prediction of data

"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=4)

import pandas as pd
import pickle

from loess.loess_1d import loess_1d
import numpy as np

from util import *
from plot_spectra_history import get_relative_spectra_histories
from generation_times_dirichlet_multinomial import get_predicted_spectrum_history

from bokeh.palettes import Dark2
from bokeh.palettes import Colorblind as cb
from bokeh.palettes import Bright

#D = Dark2[6]
#colors = [D[1], D[2], D[5], D[4], D[3], D[0]]
#colors = cb[6]
colors = Bright[6]

bins = {}
spectra = {}

for dataset in ["geva", "relate", "tsdate"]:
    s, b = get_relative_spectra_histories(dataset, "ALL")
    bins[dataset] = b
    spectra[dataset] = s

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]


def plot_spectra(dataset, ax):
    for i, c in enumerate(classes):
        y = (spectra[dataset][c] - spectra[dataset][c][0]) * 100
        x = bins[dataset]
        xout, yout, wout = loess_1d(x, y, frac=0.5, degree=2)
        ax.plot(xout, yout, c=colors[i], lw=1.0, label=c)
    ax.set_xscale("log")
    ax.set_ylabel("Percent change")
    ax.set_xlabel("Generations ago")

def plot_model_spectra(pred_arr, ax):
    for i, c in enumerate(classes):
        y = (pred_arr[:, i] - pred_arr[0, i]) * 100
        x = bins[dataset]
        xout, yout, wout = loess_1d(x, y, frac=0.5, degree=2)
        ax.plot(xout, yout, c=colors[i], lw=1.0, label=c)
    ax.set_xscale("log")
    ax.set_ylabel("Percent change")
    ax.set_xlabel("Generations ago")


fig = plt.figure(1, figsize=(6.5, 3.25))
fig.clf()
grid = (2, 3)

### top row: data spectrum histories

ax1 = plt.subplot2grid(grid, (0, 0), colspan=1, rowspan=1)
dataset = "geva"
plot_spectra(dataset, ax1)
ax1.set_title("GEVA (data)")

ax2 = plt.subplot2grid(grid, (0, 1), colspan=1, rowspan=1)
dataset = "relate"
plot_spectra(dataset, ax2)
ax2.set_title("Relate (data)")

ax3 = plt.subplot2grid(grid, (0, 2), colspan=1, rowspan=1)
dataset = "tsdate"
plot_spectra(dataset, ax3)
ax3.set_title("tsdate (data)")

ylim = max(
    [
        max(np.abs(ax1.get_ylim())),
        max(np.abs(ax2.get_ylim())),
        max(np.abs(ax3.get_ylim())),
    ]
)
ax1.set_ylim(-ylim, ylim)
ax2.set_ylim(-ylim, ylim)
ax3.set_ylim(-ylim, ylim)
ax1.set_xlabel(None)
ax2.set_xlabel(None)
ax3.set_xlabel(None)
ax2.set_ylabel(None)
ax3.set_ylabel(None)

ax1.legend(ncol=2, loc="upper left", frameon=False)

#### bottom row: model prediction spectrum histories
geva_histories = pickle.load(open("data/predicted_ages.geva.10000.pkl", "rb"))
relate_histories = pickle.load(open("data/predicted_ages.relate.10000.pkl", "rb"))
tsdate_histories = pickle.load(open("data/predicted_ages.tsdate.10000.pkl", "rb"))

iceland_spectrum = np.array([2739, 10408, 2529, 2702, 3484, 6613])

predicted_spectra_geva = get_predicted_spectrum_history(
        geva_histories["ages"]["ALL"], "geva", iceland_spectrum)
predicted_spectra_relate = get_predicted_spectrum_history(
        relate_histories["ages"]["ALL"], "relate", iceland_spectrum)
predicted_spectra_tsdate = get_predicted_spectrum_history(
        tsdate_histories["ages"]["ALL"], "tsdate", iceland_spectrum)

ax4 = plt.subplot2grid(grid, (1, 0), colspan=1, rowspan=1)
dataset = "geva"
plot_model_spectra(predicted_spectra_geva, ax4)
ax4.set_title("GEVA (model prediction)")

ax5 = plt.subplot2grid(grid, (1, 1), colspan=1, rowspan=1)
dataset = "relate"
plot_model_spectra(predicted_spectra_relate, ax5)
ax5.set_title("Relate (model prediction)")

ax6 = plt.subplot2grid(grid, (1, 2), colspan=1, rowspan=1)
dataset = "tsdate"
plot_model_spectra(predicted_spectra_tsdate, ax6)
ax6.set_title("tsdate (model prediction)")

ax4.set_ylim(-ylim, ylim)
ax5.set_ylim(-ylim, ylim)
ax6.set_ylim(-ylim, ylim)

ax5.set_ylabel(None)
ax6.set_ylabel(None)

ax4.legend(ncol=2, loc="upper left", frameon=False)

fig.tight_layout()

fig.text(0.03, 0.95, "A", fontsize=8, va="center", ha="center")
fig.text(0.36, 0.95, "B", fontsize=8, va="center", ha="center")
fig.text(0.68, 0.95, "C", fontsize=8, va="center", ha="center")
fig.text(0.03, 0.50, "D", fontsize=8, va="center", ha="center")
fig.text(0.36, 0.50, "E", fontsize=8, va="center", ha="center")
fig.text(0.68, 0.50, "F", fontsize=8, va="center", ha="center")

plt.savefig("plots/fig1.pdf")

