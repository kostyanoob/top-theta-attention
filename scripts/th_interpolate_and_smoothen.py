"""
Converts the original thresholds file (th.txt) into a th_smooth.txt where each 
threshold(seqlen) will be (1) interpolated (completing the threhsolds for missing
sequence lengths) and (2) smoothened.

Input: txt file contiaing the calibrated thresholds for all layers, heads, seqlens.
    The file is space-delimited, without headers. Each line has the pattern:
    L<layer>_H<head>:<seq_len> <threshold> <num_samples> <k>

Output: a new txt file with the same format, but each individual function has 
    been smoohtened out, and interpolated.

Usage:
    python th_txt_to_smooth_th_txt.py <PATH_TO_INPUT_th.txt>

"""
import argparse
import os
import pdb
import sys
from tqdm import trange
from typing import Tuple
sys.path.insert(0, '../')
from topk_llama import load_thresholds_from_directory
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

parser = argparse.ArgumentParser(
    description='Converts the original thresholds file (th.txt) into a th_smooth.txt where each '
                'threshold(seqlen) will be (1) interpolated (completing the threhsolds for missing'
                'sequence lengths) and (2) smoothened.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'input_file_path',
    type=str,
    help='path to the th.txt file'
)
parser.add_argument(
    '-s', '--sigma', 
    type=float,
    default=1.0,
    help='smoothening standard deviation (the higher the stronger the smoothening).'
)

args = parser.parse_args()
products_dir_path = os.path.split(args.input_file_path)[0]
output_file_path = os.path.join(products_dir_path, "th_smooth.txt")

# Read the original thresholds

def infer_num_layers_heads_from_th_txt(input_file_path) -> Tuple[int,int]:
    """
    read the input th.txt file and return the number of layers and number of heads
    depicted in it.
    """
    layers_set = set()
    heads_set = set()
    with open(args.input_file_path, 'r') as f:
        for line in f.readlines():
            layer, head_seqlen = line.split(" ")[0][1:].split("_H")
            head, seqlen = head_seqlen.split(":")
            layers_set.add(int(layer))
            heads_set.add(int(head))

    num_layers = len(layers_set)
    if max(layers_set) != len(layers_set)-1:
        raise ValueError("layer numbers in th_txt file are not enumeraed contiguously from 0 to num_layers-1")        
    num_heads = len(heads_set)
    if max(heads_set) != len(heads_set)-1:
        raise ValueError("head numbers in th_txt file are not enumeraed contiguously from 0 to num_heads-1")

    return num_layers, num_heads

def interpolate_and_smooth(th_dict, th_num_dict, M:int = None, N:int = None, sigma=1.0)-> Tuple[dict, dict]:

    if N is None:
        N = max(th_dict.keys())
    if M is None:
        M = min(th_dict.keys())

    # Extract x and f(x) values from the dictionary
    x = np.array(list(th_dict.keys()))
    fx = np.array(list(th_dict.values()))
    
    # Create an array of all x values from M to N
    x_full = np.arange(M, N + 1)
    
    # Perform linear interpolation
    interpolator = interp1d(x, fx, kind='linear', fill_value="extrapolate")
    fx_interpolated = interpolator(x_full)
    
    # Apply Gaussian smoothing
    fx_smoothed = gaussian_filter1d(fx_interpolated, sigma=args.sigma)
    
    # Create a new dictionary with the interpolated and smoothed values
    interpolated_completed_th_dictsmoothed_dict = dict(zip(x_full, fx_smoothed))

    # Complete the sample counts to be 0 for the interpolated values   
    completed_th_num_dict = {x: th_num_dict.get(x, 0) for x in interpolated_completed_th_dictsmoothed_dict.keys()}
    return interpolated_completed_th_dictsmoothed_dict, completed_th_num_dict

# load existing thresholds
num_layers, num_heads = infer_num_layers_heads_from_th_txt(args.input_file_path)
loaded_th_data_lst = []
for layer_id in trange(num_layers, desc="Loading thresholds"):
    loaded_th_data_lst.append(load_thresholds_from_directory(products_dir_path, layer_id, num_heads, verbose=False))
th_list = [d['th_list'] for d in loaded_th_data_lst]
th_num_samples = [d['th_num_samples'] for d in loaded_th_data_lst]
layerK = [d['K'] for d in loaded_th_data_lst]


# smoothen the thresholds, complete missing ones, add the num_samples for the added ones to be 0
for layer_id in trange(num_layers, desc="Interpolating and smoothing thresholds"):
    for head_id in range(num_heads):
        th_dict = th_list[layer_id][head_id]
        th_num_dict = th_num_samples[layer_id][head_id]

        # do the smoothening
        # th_dict, th_num_dict = interpolate_missing_values(th_dict, th_num_dict)
        th_dict, th_num_dict = interpolate_and_smooth(th_dict, th_num_dict, sigma=1.0)

        th_list[layer_id][head_id] = th_dict
        th_num_samples[layer_id][head_id] = th_num_dict

# write out the smoothened
with open(output_file_path,'w') as f:
    for layer_id in trange(num_layers, desc="Writing interpolated and smoothened thresholds"):
        for head_id in range(num_heads):
            for seqlen, th in th_list[layer_id][head_id].items():
                f.write(f'L{layer_id}_H{head_id}:{seqlen} {th} {th_num_samples[layer_id][head_id][seqlen]} {layerK[layer_id]}\n')

print(f'Output file written successfully to {output_file_path}')