import matplotlib.pylab as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(8765, figsize=(4, 3))
fig.clf()
ax = plt.subplot(1, 1, 1)

ax.set_xlim(0, 0.9)
ax.set_ylim(0, 1)

ax.axis("off")
ax.plot((0.1, 0.7), (0.05, 0.95), "k-", lw=1)
ax.plot((0.5, 0.3), (0.05, 0.35), "k-", lw=1)
ax.plot((0.5, 0.75), (0.65, 0.275), "k-", lw=1)
ax.plot((0.05, 0.45), (0.2, 0.8), "k-", lw=1)
ax.plot((0.45, 2 / 3), (0.8, 0.9), "k-", lw=1)

ax.annotate(
    "",
    xy=(0.05, 0.2),
    xytext=(0.2, 0.2),
    arrowprops=dict(arrowstyle="<-", linewidth=0.5),
)
ax.annotate(
    "",
    xy=(0.75, 0.275),
    xytext=(0.35, 0.275),
    arrowprops=dict(arrowstyle="<-", linewidth=0.5),
)

fig.tight_layout()
fig.subplots_adjust(left=0.05, right=1, bottom=0, top=1)

fig.text(0.15, 0.025, "W. African", fontsize=8, va="center", ha="center")
fig.text(0.575, 0.025, "European", fontsize=8, va="center", ha="center")
fig.text(0.51, 0.58, "shared", fontsize=8, va="center", ha="center", rotation=47)
fig.text(0.35, 0.58, "isolated", fontsize=8, va="center", ha="center", rotation=47)
fig.text(
    0.84, 0.24, "Neanderthal", fontsize=8, va="center", ha="center"
)  # , rotation=-47)

fig.text(0.18, 0.22, "$f$", fontsize=8, va="center", ha="center")
fig.text(0.625, 0.3, "$1-3\%$", fontsize=8, va="center", ha="center")

ax.plot((0, 0.29), (0.35, 0.35), ":", c="grey", lw=1)
fig.text(0.1, 0.365, "$\sim75$ka", fontsize=6, va="center", ha="center")

ax.plot((0, 0.49), (0.65, 0.65), ":", c="grey", lw=1)
fig.text(0.1, 0.665, "$\sim500$ka", fontsize=6, va="center", ha="center")

ax.plot((0, 0.64), (0.9, 0.9), ":", c="grey", lw=1)
fig.text(0.1, 0.915, "$>500$ka", fontsize=6, va="center", ha="center")

# Create a Rectangle patch
rect = patches.Rectangle(
    (0.15, 0.47),
    0.55,
    0.03,
    linewidth=1,
    edgecolor=None,
    facecolor="lightblue",
    linestyle="-",
)

# Add the patch to the Axes
ax.add_patch(rect)
fig.text(0.15, 0.48, "$\sim250$ka", fontsize=6, va="center", ha="center")


plt.savefig("plots/durvasula_model.pdf")
