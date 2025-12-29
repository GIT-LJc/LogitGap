
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    sns.kdeplot(-1 * ood_scores, color='#0070C0', fill=True, alpha=0.5, cut=0, clip=(0., 1.), label='OOD', legend=True)
    sns.kdeplot(-1 * id_scores, color='#55AB83', fill=True, alpha=0.5, cut=0, clip=(0., 1.), label='ID', legend=True)
    plt.ticklabel_format(axis='both', style="sci", scilimits=(0, 0))
    plt.tick_params(labelsize=16)
    plt.ylabel('Density', fontsize=20)
    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.png"))
    plt.close()


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


