#!/bin/bash
# creates sudirectories of the name format: 
#    Llama-3-8B-Instruct_arc_challenge_placement-pre-softmax_k512,512,64
# copies into each one of them an appropriate th.txt file

# Use this script to copy the th.txt files from the time-stamped directories 
# into subdirectories with meaningful parameterized names

# Specify input and output dirs
INPUT_PATH=../products
OUTPUT_PATH=../thresholds
MODEL=Llama-3.1-8B-Instruct
DATASET=arc_challenge

# Define two arrays - 
timestamp_lst=("2025-03-31_16-57-20_967957" "2025-03-31_17-44-33_761845" "2025-03-31_18-31-11_302850" "2025-03-31_16-57-28_243127" "2025-03-31_17-39-54_170189" "2025-03-31_18-16-06_639272" "2025-04-09_16-02-17_063212" "2025-03-31_16-54-20_427107" "2025-03-31_17-41-24_843294" "2025-03-31_18-27-50_955034" "2025-03-31_16-57-13_926572" "2025-03-31_17-39-33_426293" "2025-03-31_18-15-41_475172" "2025-04-09_16-12-18_607266")
k_lst=("512,512,16" "512,512,32" "512,512,64" "512,512,128" "512,512,256" "512,512,512" "756,756,756" "512,512,16" "512,512,32" "512,512,64" "512,512,128" "512,512,256" "512,512,512" "756,756,756")
placement_lst=("pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax")

# Get the length of the first list
length=${#timestamp_lst[@]}

# Iterate over both lists
for (( i=0; i<length; i++ )); do
    curr_threhsh_subdir_name=${MODEL}_${DATASET}_placement-${placement_lst[i]}_k${k_lst[i]}
    curr_output_path=$OUTPUT_PATH/$curr_threhsh_subdir_name
    mkdir -p $curr_output_path
    cp $INPUT_PATH/${timestamp_lst[i]}/th.txt $curr_output_path/
done