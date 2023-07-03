import pickle, gzip
import pandas as df
import numpy as np
import matplotlib, matplotlib.pylab as plt

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)
from bokeh.palettes import Bright

colors = Bright[6]


with gzip.open("bootstrap_dfs.pkl.gz", "rb") as fin:
    data = pickle.load(fin)

keys = list(data.keys())

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]

pop = "ALL"

counts = np.zeros((100, len(classes)))

for k in keys:
    counts += np.array(data[k][pop][classes])

for i in range(len(counts)):
    counts[i] /= counts[i].sum()

# this is the full data
y = np.zeros(counts.shape)
for i in range(len(y)):
    y[i] = (counts[i] - counts[0]) * 100


bs_data = []
num_bs = len(keys)
for j in range(num_bs):
    cs = np.random.choice(range(len(keys)), size=len(keys), replace=True)
    counts = np.zeros((100, len(classes)))
    for c in cs:
        k = keys[c]
        counts += np.array(data[k][pop][classes])
    for i in range(len(counts)):
        counts[i] /= counts[i].sum()
    y_bs = np.zeros(counts.shape)
    for i in range(len(y_bs)):
        y_bs[i] = (counts[i] - counts[0]) * 100
    bs_data.append(y_bs)

i95 = int(np.floor(num_bs * 0.025))
lower = np.zeros(y.shape)
upper = np.zeros(y.shape)
for i in range(100):
    for j in range(len(classes)):
        vals = sorted([_[i, j] for _ in bs_data])
        lower[i, j] = vals[i95]
        upper[i, j] = vals[-i95 - 1]

times = np.array((data[k][pop]["Min"] + data[k][pop]["Max"]) / 2)


fig = plt.figure(456789, figsize=(6.5, 4))
fig.clf()
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    c = classes[i]
    ax.plot(times, 0 * times, "--", color="gray", lw=0.5)
    for t, l, u in zip(times, lower[:, i], upper[:, i]):
        ax.plot((t, t), (l, u), "k-", lw=0.25)
    # ax.plot(times, lower[:, i], "--", color=colors[i])
    # ax.plot(times, upper[:, i], "--", color=colors[i])
    ax.plot(times, y[:, i], lw=1.5, color=colors[i])
    ax.set_xlabel("Generations ago")
    ax.set_ylabel("Percent change")
    ax.set_title(f"Mutation class: {c}")
    ax.set_xscale("log")

fig.tight_layout()
fig.savefig("ALL.error.pdf")
# plt.show()
