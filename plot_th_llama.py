from operator import itemgetter
import os
import pdb
from typing import Tuple
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import scipy
import tqdm
color_array = list(mcolors.TABLEAU_COLORS.values())
mpl.rcParams['agg.path.chunksize'] = 10000

parser = argparse.ArgumentParser(
    description='Reads the products directory of a single run of test_llama.py '
                'and creates the th.png and the size.png plots in the same directory.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-d', '--products_dir_path', 
    type=str, required=True,
    help='path to the products directory containing th.txt and layeri.txt files'
)
parser.add_argument(
    '-t', '--title', 
    type=str, required=True,
    help='title to be imprinted on the plots. Must contain LLaMA2-7b or LLaMA2-70b qualifier.'
)

args = parser.parse_args()

# Derive the number of layers from the title.
model_layers = {'LLaMA2-7b':32, 'LLaMA2-70b':80, 'CodeLLaMA-34b':48, 'LLaMA-3-8B':32, 'LLaMA3-70B':80}
model_heads = {'LLaMA2-7b':32, 'LLaMA2-70b':64, 'CodeLLaMA-34b':64, 'LLaMA-3-8B':32, 'LLaMA3-70B':64}
model_num_attn_layers = None
model_num_attn_heads = None
for model_name in model_layers.keys():
    if model_name.lower() in args.title.lower():
        model_num_attn_layers = model_layers[model_name]
        model_num_attn_heads = model_heads[model_name]
        break
if model_num_attn_layers is None or model_num_attn_heads is None:
    print(f"Error: the provided -run_name \"{args.title}\" must contain one of the supported models:")
    print(' ' + '\n '.join(model_layers.keys()))
    exit(-1)

#### PLOT THRESHOLD AS A FUNCTION OF SEQUENCE LENGTH - Separate subplot per layer #####
if model_num_attn_layers == 32:
    layers_to_plot = [0, 1, 10, model_num_attn_layers-1]
elif model_num_attn_layers in [48, 80]:
    layers_to_plot = [0, 5, 10, model_num_attn_layers-1]
heads_to_include = [0,4,16,20]

fig_th, ax_th = plt.subplots(len(layers_to_plot), 1, layout='constrained', sharex=True, figsize=(7,7)) 
img_filename_th=f'{args.products_dir_path}/th.png'
fig_effk, ax_effk = plt.subplots(len(layers_to_plot), 1, layout='constrained', sharex=True, figsize=(12,12)) 
img_filename_effk=f'{args.products_dir_path}/mean_effective_K_selected_threshold.png'

def plot_threshold_line(ax, seq_len_list, th_list, layer, head, legend_list, concise_legend=False):
    if len(seq_len_list) != len(th_list): 
        return legend_list
    ax.plot(seq_len_list, th_list, '.')
    legend_str = f'head {head}' if concise_legend else f'Layer-{layer} head={head} ({len(th_list)} thresholds)'
    legend_list.append(legend_str)
    return legend_list

def plot_sample_count_line(ax, seq_len_list, samples_list, color='purple'):
    ax.set_ylabel('Calib. samples', color = color) 
    ax.bar(seq_len_list, samples_list, color = color, label='Num. calibration samples', alpha=0.5)  
    ax.tick_params(axis ='y', labelcolor = color)  
    ax.legend(loc='upper left')

input_path = f'{args.products_dir_path}/th.txt'
if os.path.exists(input_path):
    print(f"Plotting calibrated thresholds as a function of sequence length ({img_filename_th})")
    print(f"Plotting mean effective K (calibration set) of the selected threshold as a function of sequence length ({img_filename_effk})")
    with open(input_path, "r") as f:
        layer=0
        head=0
        subplot_id=0
        legend_list_th=[]
        seq_len_list=[]
        th_list=[]
        samples_list=[]
        mean_effective_K_lst=[]
        legend_list_effk=[]
        lines = f.readlines() # each line should contain at least "L:<layerid> seq_len threshold"
        for linenum, line in enumerate(lines):

            # Parse the th.txt's line
            line_layer_head, line = line.split(':')
            line_vals = line.split(' ')
            if len(line_vals) == 2:
                seq_len, th, samples = int(line_vals[0]), float(line_vals[1]), None
            elif len(line_vals) == 3:
                seq_len, th, samples = int(line_vals[0]), float(line_vals[1]), int(line_vals[2])
            elif len(line_vals) == 4:
                seq_len, th, samples, mean_effective_K = int(line_vals[0]), float(line_vals[1]), int(line_vals[2]), float(line_vals[3])        
            if len(seq_len_list) == 0 or line_layer_head==f"L{layer}_H{head}":
                # Continue gathering thresholds for the current layer
                seq_len_list.append(seq_len)
                th_list.append(th)
                samples_list.append(samples)
                mean_effective_K_lst.append(mean_effective_K)
            
            if line_layer_head !=f"L{layer}_H{head}" or linenum == len(lines)-1:
                # Finished reading thresholds of one layer
                # Plot threshold values of this layer-head (if requested to plot)
                if layer in layers_to_plot and head in heads_to_include:
                    # plot threshold
                    plot_threshold_line(ax_th[subplot_id], seq_len_list, th_list, layer, head, legend_list_th, concise_legend=True)
                    # plot effective K on a separate plot
                    plot_threshold_line(ax_effk[subplot_id], seq_len_list, mean_effective_K_lst, layer, head, legend_list_effk, concise_legend=True)

                    # plot the number of samples only once and finalize the current subplot
                    if head == heads_to_include[-1] and all(map(lambda s:s is not None, samples_list)):
                        # th plot
                        plot_sample_count_line(ax_th[subplot_id].twinx() , seq_len_list, samples_list)
                        ax_th[subplot_id].set_title(f'Layer {layer}')
                        ax_th[subplot_id].set_xlabel('Sequence Length')
                        ax_th[subplot_id].set_ylabel('Threshold')
                        ax_th[subplot_id].grid()
                        ax_th[subplot_id].legend(legend_list_th, loc='lower right', ncols=1, framealpha=0.5)   #ncols=len(legend_list_th)   
                        legend_list_th=[]        

                        # make y-axis longer to accommodate the legend - do it for every subplot
                        ymin, ymax = ax_th[subplot_id].get_ylim()
                        ax_th[subplot_id].set_ylim(ymin, ymax + (ymax-ymin) * 0.25)

                        # mean_effective_K_selected_threshold plot
                        ax_effk[subplot_id].set_title(f'Layer {layer}')
                        ax_effk[subplot_id].set_xlabel('Sequence Length')
                        ax_effk[subplot_id].set_ylabel('Mean Effective K')
                        ax_effk[subplot_id].grid()
                        ax_effk[subplot_id].legend(legend_list_effk, loc='best', framealpha=1)   
                        legend_list_effk = []

                        subplot_id += 1        
                
                # Prepare to parse next head or layer
                if linenum < len(lines)-1:
                    if head < model_num_attn_heads - 1:
                        head += 1
                    else:
                        head = 0
                        layer += 1                             
                    seq_len_list = [seq_len]
                    th_list = [th]
                    samples_list=[samples]
                    mean_effective_K_lst=[mean_effective_K]
    
    # make x-axis longer to accommodate the legend - do it once, since all subplots share the x axis
    xmin, xmax = ax_th[0].get_xlim()
    ax_th[0].set_xlim(xmin, xmax + (xmax-xmin) * 0.25)    

    # Write the plot to file "th"           
    fig_th.suptitle(args.title)
    fig_th.savefig(img_filename_th, dpi=300)
    plt.close(fig_th)

    # Write the plot to file "mean_effective_K_selected_threshold"
    fig_effk.suptitle(args.title + '\n computed over calibration samples')
    fig_effk.savefig(img_filename_effk, dpi=300)
    plt.close(fig_effk)
else:
    print(f"Warning: {input_path} was not found. Plots 'th.png' and 'mean_effective_K_selected_threshold.png' will not be produced.")




#### PLOT BUFFER SIZE INCREASE FACTOR (TH compared to top-k) AS A FUNCTION OF SEQUENCE LENGTH #####
layers_to_plot = [0, 10, model_num_attn_layers-1]
heads_to_include = list(range(model_num_attn_heads)) # or "ALL" for older versions where no per-head information 
layer = 0
legend_list_th = []
img_filename = f'{args.products_dir_path}/size.png'
smoothing_win_size = 19
warning_flag = False
if all(os.path.exists(f'{args.products_dir_path}/layer{l}.txt') for l in layers_to_plot):
    print(f"Plotting memory size increase comparing to Top-k ({img_filename})")
    for subplot_id,l in enumerate(layers_to_plot):
        sz_list=[]
        len_list=[]
        with open(f'{args.products_dir_path}/layer{l}.txt', 'r') as f:
            for line in f.readlines():
                l_str, k, comment, sz_factor = line.split(' ')
                lid, seq_len = l_str.split(':')
                if lid.startswith(f"L{l}"):
                    lid_split = lid.split("_H")
                    if len(lid_split) != 2 and not warning_flag:
                        warning_flag = True
                        print("Warning: Old format of layer.txt: lines are not prefixed by L<layer>_H<head>. Will include all heads in the analysis")
                    if len(lid_split) != 2 or heads_to_include == "ALL" or int(lid_split[1]) in heads_to_include:
                        sz_list.append(float(sz_factor))
                        len_list.append(int(seq_len))
        
        plt.plot(len_list, sz_list, color=color_array[subplot_id], alpha=0.3)
        legend_list_th.append(f'Layer-{l}')
        plt.plot(len_list, scipy.signal.savgol_filter(sz_list, smoothing_win_size, 1, mode='nearest'), '--', color=color_array[subplot_id])    
        legend_list_th.append(f'Layer-{l} (avg)')

    plt.title(f'{args.title}\nMemory Size Comparison Top-K and Thresholding (evaluation set)')
    plt.ylabel('Size Factor TH / TOP-K')
    plt.xlabel('Sequence Length')
    plt.legend(legend_list_th, loc='best')
    plt.tight_layout()
    plt.savefig(img_filename)
    plt.close()
    plt.show()
else:
    print(f"Warning: not all the files of the form {args.products_dir_path}/layer<layernum>.txt exist. Skipping the size plots.")




# #### PLOT Effective K (in calibration set) as a function of threshold - separate subplot per layer #####
if model_num_attn_layers == 32:
    layers_to_plot = [0, 1, 10, model_num_attn_layers-1]
elif model_num_attn_layers in [48, 80]:
    layers_to_plot = [0, 5, 10, model_num_attn_layers-1]
heads_to_plot = [0,4,16,20]
seq_lens_to_plot = [800]  #[65, 128,256,513,756,1024,1256]
layer_to_ylims = [(412,612) if l in [0,1] else (52,72) for l in layers_to_plot] # rage of K's *y-axes) to zoom into when plotting the zoomed-in plot

fig_effk_all, ax_effk_all = plt.subplots(len(layers_to_plot), 1, layout='constrained', sharex=True, figsize=(12,12)) 
fig_effk_all_zoomin, ax_effk_all_zoomin = plt.subplots(len(layers_to_plot), 1, layout='constrained', sharex=True, figsize=(12,12)) 

img_filename_effk_all=f'{args.products_dir_path}/mean_effective_K_all_threholds.png'
img_filename_effk_all_zoomin=f'{args.products_dir_path}/mean_effective_K_all_threholds_zoomin.png'

def plot_effk_all_line(ax, th_lst, effective_k_lst, layer, head, seq_len, concise_legend=False) -> None:
    if len(seq_len_list) != len(th_list): 
        return
    legend_str = f'head {head} seqlen {seq_len}' if concise_legend else f'Layer-{layer} head={head} seqlen={seq_len} ({len(th_list)} thresholds)'    
    ax.plot(th_lst, effective_k_lst, '.', label=legend_str)

def get_calibrated_th_effk(th_txt_path:str, layer:int, head:int, seq_len:int) -> Tuple[float, float]:
    """
    Read the calibrated threshold and the corresponding effective k (effective 
    on the calibration set) for the specified (layer, head, seqlen)
    """
    with open(th_txt_path) as f:
        for line in tqdm.tqdm(f.readlines(), desc="Parsing lines"):
            # one line looks like:
            # L31_H31:1219 0.0002257227897644043 4 64.0
            header, th, num_samples, effk = line.split(" ")
            if header == f"L{layer}_H{head}:{seq_len}":
                return float(th), float(effk)
    return None, None
            
def plot_cross(ax, x_cross:float, y_cross:float, cross_height=0.05, cross_width=0.0125):
    """
    Superimpose a purple cross on the axes "ax" at the specified x,y coordinates
    """
    if x_cross is not None and y_cross is not None:
        cross_len_horizontal = (ax.get_xlim()[1] - ax.get_xlim()[0]) * cross_width
        cross_len_vertical = (ax.get_ylim()[1] - ax.get_ylim()[0]) * cross_height
        ax.plot([x_cross - cross_len_horizontal, x_cross + cross_len_horizontal], [y_cross, y_cross], linestyle='-', color='purple', alpha=0.7, label='')
        ax.plot([x_cross, x_cross], [y_cross - cross_len_vertical, y_cross + cross_len_vertical], linestyle='-', color='purple', alpha=0.7, label='')

# Read the huge effectiveK.txt file and plot the relevant layers, heads, seq-lens into 2 figures (zoomed-out and zoomed-in)
input_path = f'{args.products_dir_path}/th_effectiveK.txt'
if os.path.exists(input_path):
    print(f"Plotting mean effective K (calibration set) of a range of thresholds as a function of sequence length ({img_filename_effk_all})")
    with open(input_path) as f:
        for line in tqdm.tqdm(f.readlines(), desc="Parsing lines"):
            # Parse the th_effectiveK.txt's line
            # L0_H0:1235 th>mean_eff_K 0.01>64,0.001>65,0.0001>66\n
            header, comment_str, payload = line[1:].split(" ")
            layer, head, seq_len = map(int, header.replace("_H"," ").replace(":", " ").split(" "))
            if layer in layers_to_plot and head in heads_to_plot and seq_len in seq_lens_to_plot:
                subplot_id = layers_to_plot.index(layer) 
                
                # read the actual threshold that was eventually calibrated
                th_calib, effk_calib = get_calibrated_th_effk(f'{args.products_dir_path}/th.txt', layer, head, seq_len) 
                if th_calib is None or effk_calib is None:
                    print("Warning: could not locate the selected threshold for layer {layer}, head {head}, seq_len {seq_len}. A corss is not going to be rendered.")

                # parse the payload string that has the pattern of "th1>k1,th2>k2,...."
                th_effk_pairs_lst = list(map(lambda tup: (float(tup[0]),float(tup[1])), map(lambda s: s.split(">"), payload.split(","))))
                th_lst = list(map(itemgetter(0), th_effk_pairs_lst))
                effective_k_lst = list(map(itemgetter(1), th_effk_pairs_lst))
                plot_effk_all_line(ax_effk_all[subplot_id], th_lst, effective_k_lst, layer, head, seq_len, concise_legend=True)
                plot_cross(ax_effk_all[subplot_id], th_calib, effk_calib)

                # plot the thresholds at zoomed-in view (to the interesting k range)
                th_effk_pairs_lst_zoomin = list(filter(lambda th_effk: layer_to_ylims[subplot_id][0] <= th_effk[1] <= layer_to_ylims[subplot_id][1], th_effk_pairs_lst))
                th_lst_zoomin = list(map(itemgetter(0), th_effk_pairs_lst_zoomin))
                effective_k_lst_zoomin = list(map(itemgetter(1), th_effk_pairs_lst_zoomin))
                plot_effk_all_line(ax_effk_all_zoomin[subplot_id], th_lst_zoomin, effective_k_lst_zoomin, layer, head, seq_len, concise_legend=True)
                plot_cross(ax_effk_all_zoomin[subplot_id], th_calib, effk_calib)

        # Finalize plots
        for subplot_id, layer in enumerate(layers_to_plot):

            # zoomed-out plot
            ax_effk_all[subplot_id].set_title(f'Layer {layer}')
            ax_effk_all[subplot_id].set_xlabel('Threshold')
            ax_effk_all[subplot_id].set_ylabel('Mean Effective K')
            ax_effk_all[subplot_id].grid()
            handles, labels = ax_effk_all[subplot_id].get_legend_handles_labels()
            handles_to_show = [h for h,l in zip(handles, labels) if l!='']
            labels_to_show = [l for l in labels if l!='']
            ax_effk_all[subplot_id].legend(handles_to_show, labels_to_show, loc='best', framealpha=1)   
            
            # zoomed-in plot
            ax_effk_all_zoomin[subplot_id].set_title(f'Layer {layer}')
            ax_effk_all_zoomin[subplot_id].set_xlabel('Threshold')
            ax_effk_all_zoomin[subplot_id].set_ylabel('Mean Effective K')
            ax_effk_all_zoomin[subplot_id].grid()
            handles, labels = ax_effk_all_zoomin[subplot_id].get_legend_handles_labels()
            handles_to_show = [h for h,l in zip(handles, labels) if l!='']
            labels_to_show = [l for l in labels if l!='']        
            ax_effk_all_zoomin[subplot_id].legend(handles_to_show, labels_to_show, loc='best', framealpha=1)   

    # save zoomed-out plot
    fig_effk_all.suptitle(args.title + '\n computed over calibration samples')
    fig_effk_all.savefig(img_filename_effk_all)
    plt.close(fig_effk_all)

    # save zoomed-in plot
    fig_effk_all_zoomin.suptitle(args.title + '\n computed over calibration samples')
    fig_effk_all_zoomin.savefig(img_filename_effk_all_zoomin)
    plt.close(fig_effk_all_zoomin)
else:
    print(f"Warning: file {input_path} does not exist, skipping the ploting of 'mean_effective_K_all_threholds' plots.")
