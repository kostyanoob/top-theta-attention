from typing import Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import os

METRIC_DESCRIPTION = {"acc_norm": "Accuracy (normalized)", 
                      "pass@1": "pass@1", 
                      "kept_attn": "Attention elements",
                      "kept_vrow": "V rows"}

def plot_accuracy_vs_size(df:pd.DataFrame, 
                          title:str, 
                          output_dir_path:str, 
                          full_xlim:bool=True,
                          full_ylim:bool=True,
                          normalize_size_by_fullsize:bool=False,
                          x_label_prefix:str="",
                          plot_type:str='errorbar',
                          accuracy_jitter:Union[bool,float]=False,
                          different_markers:bool=False,
                          rcParams_override:dict={},
                          concise_labels:bool=False):
    """
    Renders scatterplot showing accuracy (Y-axis) vs size (X-axis) of several models described in the dataframe "df"
    and groups the plotted dots by the "label" column of the dataframe.

    Args:
    df: pd.DataFrame - from which size_mean will be used for X and accuracy_mean will be used for Y
    title: str - to show as a plot title
    output_dir_path: str - directory where the image should be saved
    full_xlim: bool - True means showing X range from 0 to 1.0 without restricting to relevant range
    full_ylim: bool - True means showing Y range from 0 to 1.0 without restricting to relevant range
    normalize_size_by_fullsize: bool - whether the X value should b normalized to 1 by dividing size_mean by fullsize_mean
    x_label_prefix:string to prepend before the X-axis title
    plot_type: can be one of the following:
                - "errorbar": to plot scatter plot with error bars
                - "line": plot line for every label, without errorbars
    accuracy_jitter:bool or float 
                  - if True - adds a 0.1 hight jitter to every y value (accuracy) which is not unique - use to resolve overlapping of points and error-bars to improve readability 
                  - Can also be set to a fraction in the range (0.0, 1.0) to override the default 0.1 jittering step (the larger the stronger the jitter)
    different_markers: bool - True:  use different markers for different series labels; False: use dot marker
    rcParams_override: dictionary with pyplot.rcParams to be overrided
    concise_labels:bool - shorten the legend labels to contain only words before the first symbol of '+'
    """
    
    # Default image parameters
    rcParams_custom = {
        'figure.figsize' : [6, 3],
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 8
    }
    # Override the rcParams of the figure with what the user provided
    rcParams_custom.update(rcParams_override)

    plt.style.use('default')
    plt.rcParams.update(rcParams_custom)
    markers = ['o', 'v', '^','<','>'] if different_markers else ['o']

    if normalize_size_by_fullsize:
        # normalize size_mean by fullsize_mean
        df = df.copy()
        df.size_mean = df.size_mean.divide(df.fullsize_mean)
        df.size_std = df.size_std.divide(df.fullsize_mean)

    if accuracy_jitter == True or accuracy_jitter > 0:
        # Add jitter only to non-unique y values
        accuracy_jitter_size = 0.01 if accuracy_jitter else accuracy_jitter
        df = df.copy()
        jitter_amount = (df['accuracy_mean'].max() - df['accuracy_mean'].min()) * accuracy_jitter_size
        non_unique_y = df['accuracy_mean'].duplicated(keep=False)
        df.loc[non_unique_y, 'accuracy_mean'] += np.random.uniform(-jitter_amount, jitter_amount, non_unique_y.sum())

    if concise_labels:
        df = df.copy()
        df['label'] = df['label'].str.split('+').str[0]

    fig, ax = plt.subplots()

    if plot_type == 'line':
        for i,label in enumerate(df['label'].unique()):
            subset = df[df['label'] == label]
            ax.plot(subset['size_mean'], 
                    subset['accuracy_mean'], 
                    marker=markers[i%len(markers)],
                    label=label)
    elif plot_type == 'errorbar':
        for i,label in enumerate(df['label'].unique()):
            subset = df[df['label'] == label]
            plt.errorbar(
                x=subset['size_mean'],
                y=subset['accuracy_mean'],
                xerr=subset[f'size_std'],
                yerr=subset[f'accuracy_std'],
                fmt=markers[i%len(markers)], #'o',
                label=label,
                capsize=5,
                alpha=0.7
            )
    else:
        print(f"Error: plot_type={plot_type} is not supported. Please choose either 'errorbar' or 'line'")

    accuracy_metric = df.accuracy_metric.unique().item()
    size_metric = df.size_metric.unique().item()
    
    plt.xlabel(f'{x_label_prefix} {METRIC_DESCRIPTION[size_metric]}')
    plt.ylabel(METRIC_DESCRIPTION[accuracy_metric])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if full_xlim:
        ax.set_xlim(0.0, 1.05)
    
    if full_ylim:
        ax.set_ylim(0.0, 1.0)

    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0, symbol='%'))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0 if (df.accuracy_mean.max()-df.accuracy_mean.min() > 0.02) else 1, symbol='%'))

    # Save image to file
    plt.tight_layout()    
    os.makedirs(output_dir_path, exist_ok=True)
    title_clean = title.replace(' ','_').replace('(','').replace(')','').replace(' - ',' ')
    plt.savefig(f"{output_dir_path}/{title_clean}.png")
    plt.savefig(f"{output_dir_path}/{title_clean}.eps")

    # Show image on the notebook
    plt.title(title) # title won't appear in the saved file on purpose
    plt.show()