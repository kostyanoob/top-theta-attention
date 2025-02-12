import argparse
import os
import pandas as pd
import json
import pdb
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
color_array = list(mcolors.TABLEAU_COLORS.values())
mpl.rcParams['agg.path.chunksize'] = 10000

parser = argparse.ArgumentParser(
    description="Reads the products of the generative HumanEval test, creates various plots for it, writes the files within the same directory",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-d',
    '--products_dir_path',
    type=str, required=True,
    help='Path to the products directory of the run ')
parser.add_argument(
    '-t', '--title', 
    type=str, required=True,
    help='title to be imprinted on the plots. Must contain LLaMA2-7b or LLaMA2-70b qualifier.'
)

args = parser.parse_args()

DPI=300

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

# Read samples.jsonl_results.jsonl
with open(args.products_dir_path + "/samples.jsonl_results.jsonl") as json_file:
    results_dicts_lst = [json.loads(line.strip()) for line in json_file]
    results_df = pd.DataFrame(results_dicts_lst)


# Read prompt_completion_lengths_per_sample.csv
seq_len_df = pd.read_csv(args.products_dir_path+"/prompt_completion_lengths_per_sample.csv", 
			 delimiter=" ", 
			 header=None, 
			 names=["seq_len_prompt", "seq_len_completion"])

# concatenate the two dataframes
df = pd.concat([results_df, seq_len_df], axis=1)
df['seq_len_total'] = df.seq_len_prompt + df.seq_len_completion

# Plot dataset sequence lengths
img_filename = args.products_dir_path + '/prompt_lengths.png'
print(f"Plotting dataset (prompt) sequence lengths histogram ({img_filename})")
ax = df[['task_id','seq_len_prompt']].drop_duplicates().seq_len_prompt.plot.hist(bins=30)
ax.set_ylabel(f"Number of prompts (total {df.task_id.nunique()})")
ax.set_xlabel(f"Prompt length in tokens (average {df.seq_len_prompt.mean():.1f})")
ax.set_title("Prompt Sequence Lengths")
fig = ax.get_figure()
fig.savefig(img_filename, dpi=DPI)
plt.close(fig)

# Plot completion sequence lengths
img_filename = args.products_dir_path + '/completion_lengths.png'
print(f"Plotting completion sequence lengths histogram ({img_filename})")
passed = df[df['passed']==True]['seq_len_completion']
failed = df[df['passed']==False]['seq_len_completion']
plt.hist([passed, failed], bins=30, 
         stacked=True, edgecolor='black', linewidth=0, 
         color=['green', 'red'], label=['Passed', 'Failed'])
ax = plt.gca()
ax.set_ylabel(f'Number of completion samples (total {len(passed)+len(failed)})')
ax.set_xlabel(f'Completion sequence length [tokens] (average {df.seq_len_completion.mean():.1f})')
ax.set_title('Completion lengths of the samples')
ax.legend()
fig = ax.get_figure()
fig.savefig(img_filename, dpi=DPI)
plt.close(fig)

# plot total sequence lengths (total=prompt+completion)
img_filename = args.products_dir_path + '/total_lengths.png'
print(f"Plotting total sequence lengths histogram ({img_filename})")
passed = df[df['passed']==True]['seq_len_total']
failed = df[df['passed']==False]['seq_len_total']
plt.hist([passed, failed], bins=30,
         stacked=True, edgecolor='black', linewidth=0, 
         color=['green', 'red'], label=['Passed', 'Failed'])
ax = plt.gca()
ax.set_ylabel(f'Number of completion samples (total {len(passed)+len(failed)})')
ax.set_xlabel(f'Total sequence length [tokens] (average {df.seq_len_total.mean():.1f})')
ax.set_title('Total sequence lengths (prompt+completion) of the samples')
ax.legend()
fig = ax.get_figure()
fig.savefig(img_filename, dpi=DPI)
plt.close(fig)



#### PLOT BUFFER SIZE INCREASE FACTOR (TH compared to top-k) AS A FUNCTION OF SEQUENCE LENGTH #####
# Lines of Layer{i}.txt files looks as follows:
# L1_H61:129 512 prefill 1.0
# L1_H62:129 512 prefill 1.0
# L1_H63:129 512 prefill 1.0
# L1_H0:130 512 generative_decoding 0.007692307699471712
# L1_H1:130 512 generative_decoding 0.007692307699471712
# L1_H2:130 512 generative_decoding 0.007692307699471712
layers_to_plot = [0, 10, model_num_attn_layers-1]
heads_to_include = list(range(model_num_attn_heads)) # or "ALL" for older versions where no per-head information 
for inference_phase in ["prefill", "generative_decoding"]:
    layer = 0
    legend_list_th = []
    img_filename = f'{args.products_dir_path}/size_{inference_phase}.png'
    smoothing_win_size = 19
    warning_flag = False
    if all(os.path.exists(f'{args.products_dir_path}/layer{l}.txt') for l in layers_to_plot):
        print(f"Plotting memory size increase comparing to Top-k ({img_filename})")
        for subplot_id,l in enumerate(layers_to_plot):
            sz_list=[]
            len_list=[]
            with open(f'{args.products_dir_path}/layer{l}.txt', 'r') as f:
                for line in f.readlines():
                    l_str, k, inf_phase, sz_factor = line.split(' ')
                    if inf_phase != inference_phase:
                        continue
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

        plt.title(f'{args.title}\nMemory Size Comparison Top-K and Thresholding ({inference_phase}, evaluation set)')
        plt.ylabel('Size Factor TH / TOP-K')
        plt.xlabel('Sequence Length')
        plt.legend(legend_list_th, loc='best')
        plt.tight_layout()
        plt.savefig(img_filename)
        plt.close()
        plt.show()
    else:
        print(f"Warning: not all the files of the form {args.products_dir_path}/layer<layernum>.txt exist. Skipping the size plots.")

