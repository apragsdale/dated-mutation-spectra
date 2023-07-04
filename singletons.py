import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)
from bokeh.palettes import Bright

colors = Bright[6]
from loess.loess_1d import loess_1d


pop = "ALL"

relate_singletons = pd.read_csv(
    "data/binned_ages.relate.ALL.max_age.10000.only_singletons.csv", sep="\t"
)
tsdate_singletons = pd.read_csv(
    "data/binned_ages.tsdate.ALL.max_age.10000.only_singletons.csv", sep="\t"
)

relate_non = pd.read_csv(
    "data/binned_ages.relate.ALL.max_age.10000.singletons.False.max_frequency.0.98.csv",
    sep="\t",
)
tsdate_non = pd.read_csv(
    "data/binned_ages.tsdate.ALL.max_age.10000.singletons.False.max_frequency.0.98.csv",
    sep="\t",
)

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]

y_s_r = np.array([[relate_singletons[c][i] for c in classes] for i in range(100)])
y_s_t = np.array([[tsdate_singletons[c][i] for c in classes] for i in range(100)])

y_n_r = np.array([[relate_non[c][i] for c in classes] for i in range(100)])
y_n_t = np.array([[tsdate_non[c][i] for c in classes] for i in range(100)])

times_r_s = np.array(relate_singletons["Min"] + relate_singletons["Max"]) / 2
times_t_s = np.array(tsdate_singletons["Min"] + tsdate_singletons["Max"]) / 2
times_r_n = np.array(relate_non["Min"] + relate_non["Max"]) / 2
times_t_n = np.array(relate_non["Min"] + relate_non["Max"]) / 2

num_r_sing = np.sum(y_s_r)
num_t_sing = np.sum(y_s_t)
num_r_non = np.sum(y_n_r)
num_t_non = np.sum(y_n_t)

fig = plt.figure(13579, figsize=(6.5, 2.5))
fig.clf()

ax1 = plt.subplot(1, 3, 1)

ax1.plot(times_r_s, y_s_r.sum(axis=1) / num_r_sing, label="Relate")
ax1.plot(times_t_s, y_s_t.sum(axis=1) / num_t_sing, label="tsdate")
ax1.set_xscale("log")
ax1.set_xlabel("Time ago (generations)")
ax1.set_ylabel("Proportion singletons")
ax1.legend()

y_s_r = (y_s_r.T / y_s_r.sum(axis=1)).T
y_s_t = (y_s_t.T / y_s_t.sum(axis=1)).T
y_n_r = (y_n_r.T / y_n_r.sum(axis=1)).T
y_n_t = (y_n_t.T / y_n_t.sum(axis=1)).T

ax2 = plt.subplot(1, 3, 2)

for j, c in enumerate(classes):
    y = 100 * (y_s_r[:, j] - y_n_r[0, j])
    ax2.plot(times_r_s, y, c=colors[j], lw=0.1, label=None)
for j, c in enumerate(classes):
    y = 100 * (y_s_r[:, j] - y_n_r[0, j])
    xout, yout, wout = loess_1d(times_r_s, y, frac=0.5, degree=2)
    ax2.plot(times_r_s, yout, c=colors[j], lw=2.0, label=c)

ax2.set_xscale("log")
ax2.legend(ncol=2, fontsize=5)
ax2.set_title("Relate singletons")
ax2.set_ylabel("Percent change")
ax2.set_xlabel("Generations ago")

ax3 = plt.subplot(1, 3, 3)

for j, c in enumerate(classes):
    y = 100 * (y_s_t[:, j] - y_n_t[0, j])
    ax3.plot(times_t_s, y, c=colors[j], lw=0.1, label=None)
for j, c in enumerate(classes):
    y = 100 * (y_s_t[:, j] - y_n_t[0, j])
    xout, yout, wout = loess_1d(times_t_s, y, frac=0.5, degree=2)
    ax3.plot(times_t_s, yout, c=colors[j], lw=2.0, label=c)

ax3.set_xscale("log")
ax3.set_title("tsdate singletons")
ax3.set_xlabel("Generations ago")

fig.tight_layout()
plt.savefig("plots/singletons.pdf")
