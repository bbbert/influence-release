from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

def plot_z_norms(ax, z_norms, dataset_id,
                 title="Distribution of Z-norms",
                 subtitle=None):
    ax.hist(z_norms, bins=50)
    ax.set_xlabel('Z-norm')
    ax.set_ylabel('Frequency')

    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)

def generate_color_cycle(labels):
    unique_labels = np.unique(labels)
    unique_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, unique_colors))
    return [label_to_color[label] for label in labels], label_to_color

def plot_influence_correlation(ax,
                               actl,
                               pred,
                               label=None,
                               title="Predicted against actual influence",
                               subtitle="None",
                               xlabel="Actual influence",
                               ylabel="Predicted influence",
                               balanced=False):
    # Compute data bounds
    if balanced:
        maxW = max(np.max(np.abs(actl)), np.max(np.abs(pred)))
        minW = -maxW
    else:
        maxW = max(np.max(actl), np.max(pred))
        minW = min(np.min(actl), np.min(pred))

    # Expand bounds
    padding = 0.05 * (maxW - minW)
    minW, maxW = minW - padding, maxW + padding

    # Plot x=y
    ax.plot([minW, maxW], [minW, maxW], color='grey', alpha=0.3)

    # Color groups of points if tagged
    colors = None
    if label is not None:
        colors, label_to_color = generate_color_cycle(label)

        # Plot dummy points
        for label, color in label_to_color.items():
            ax.scatter(maxW + 10, maxW + 10, color=color, label=label, alpha=0.5, s=10)

    # Plot points
    ax.scatter(actl, pred, color=colors, alpha=0.5, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([minW, maxW])
    ax.set_ylim([minW, maxW])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)
