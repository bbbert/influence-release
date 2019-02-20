import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
sns.set(color_codes=True)

import scipy.stats

def plot_dataset(ax, X, y, annotations=[], title="Dataset",
                 grid_function=None, grid_samples=[50, 50],
                 grid_cmap=sns.cubehelix_palette(dark=.9, light=.2, as_cmap=True)):
    if title is not None:
        ax.set_title(title)
    
    W = np.max(np.abs(X))
    
    if grid_function is not None:
        NX, NY = grid_samples
        grid_X, grid_Y = np.meshgrid(np.linspace(-W, W, NX + 1), np.linspace(-W, W, NY + 1))
        X_sample = np.vstack((((grid_X[:-1, :-1] + grid_X[1:, 1:]) / 2).reshape(-1),
                              ((grid_Y[:-1, :-1] + grid_Y[1:, 1:]) / 2).reshape(-1))).T
        samples = np.array([grid_function(x) for x in X_sample]).reshape(NX, NY)
        
        c = ax.pcolormesh(grid_X, grid_Y, samples, cmap=grid_cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(c, cax=cax, orientation='vertical')
    
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap('bwr')
    ax.set_xlim([-W, W])
    ax.set_ylim([-W, W])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm, s=2)
    ax.set_aspect('equal')
    
    for x, y, label in annotations:
        ax.scatter(x[0], x[1], c=y, cmap=cmap, norm=norm, s=5)
        ax.annotate(label, xy=x, xytext=(-40, 20),
            textcoords='offset points',
            arrowprops=dict(facecolor='black', shrink=0.05),
            bbox=dict(boxstyle="square,pad=0.3", fc='white', ec='black', lw=1))

def plot_decision_boundary(ax, d):
    d = d / np.linalg.norm(d)
    
    line = 100 * np.array([[-d[1], d[0]], [d[1], -d[0]]])
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(line[:, 0], line[:, 1], color='green', linewidth=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.quiver(0, 0, d[0], d[1], color='green')

def plot_hessian(ax, H, origin=None):
    origin = origin if origin is not None else np.zeros(H.shape[0])
    
    eigvals, eigs = np.linalg.eig(H)
    scale = 1.0 / np.max(np.abs(eigvals))
    angle = np.rad2deg(np.arctan2(eigs[1, 0], eigs[0, 0]))
    
    ax.quiver(origin[0], origin[1], eigs[0, :], eigs[1, :], color='purple')
    
    eigvals *= scale
    ellipse = patches.Ellipse(origin, width=eigvals[0], height=eigvals[1], angle=angle,
                              linewidth=1, fill=None, color='purple')
    ax.add_patch(ellipse)

def plot_lines(ax, x, lines, title=None):
    if title is not None:
        ax.set_title(title)
    
    for y, label in lines:
        y = np.full(x.shape, y)
        ax.plot(x, y, label=label)

    ax.legend()
    
def plot_correlation(ax, x, y,
                     xlabel="Actual influence", ylabel="Predicted influence",
                     title="Actual over predicted",
                     equal=True, balanced=True, expand=True,
                     draw_line=True):
    xlim = np.array([np.min(x), np.max(x)])
    ylim = np.array([np.min(y), np.max(y)])
    if balanced:
        xlim = np.array([-1, 1]) * np.max(np.abs(xlim))
        ylim = np.array([-1, 1]) * np.max(np.abs(ylim))
    if expand:
        xlim = xlim + np.array([-1, 1]) * (xlim[1] - xlim[0]) * 0.1
        ylim = ylim + np.array([-1, 1]) * (ylim[1] - ylim[0]) * 0.1
        
    abslim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    if equal:
        xlim = ylim = abslim
    
    rho, p = scipy.stats.spearmanr(x, y)
    
    ax.scatter(x, y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("{} (rho={},p={})".format(title, rho, p))
    
    
    
    if draw_line:
        ax.plot(abslim, abslim)