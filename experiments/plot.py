from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    # Plot points
    ax.scatter(actl, pred, color=colors, alpha=0.5, s=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([minW, maxW])
    ax.set_ylim([minW, maxW])

    legend_elements = [ Line2D([0], [0], linewidth=0, marker='o',
                               color=label_color, label=label, markersize=5)
                        for label, label_color in label_to_color.items() ]
    ax.legend(handles=legend_elements,
              loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)

def plot_against_subset_size(ax,
                             subset_tags,
                             subset_indices,
                             value,
                             title='Group self-influence against subset size',
                             xlabel='Group size',
                             ylabel='Influence',
                             subtitle=None):
    subset_sizes = np.array([len(indices) for indices in subset_indices])
    maxS = np.max(subset_sizes)
    maxV = np.max(value)

    label = subset_tags
    colors, label_to_color = generate_color_cycle(label)

    for label, label_color in label_to_color.items():
        cur_subsets = np.array(subset_tags) == label
        cur_sizes = subset_sizes[cur_subsets]
        cur_values = np.array(value)[cur_subsets]
        sort_idx = np.argsort(cur_sizes)
        cur_sizes = cur_sizes[sort_idx]
        cur_values = cur_values[sort_idx]
        ax.plot(cur_sizes, cur_values, c=label_color, label=label,
                alpha=0.5, marker='o', markersize=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    if subtitle is not None:
        title = title + "\n" + subtitle
    ax.set_title(title)
