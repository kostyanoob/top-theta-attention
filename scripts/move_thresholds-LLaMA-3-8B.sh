#!/bin/bash
# creates sudirectories of the name format: 
#    Llama-3-8B-Instruct_arc_challenge_placement-pre-softmax_k512,512,64
# copies into each one of them an appropriate th.txt file

# Use this script to copy the th.txt files from the time-stamped directories 
# into subdirectories with meaningful parameterized names

# Specify input and output dirs
INPUT_PATH=../products
OUTPUT_PATH=../thresholds
MODEL=Llama-3-8B
DATASET=arc_challenge

# Define two arrays - 
timestamp_lst=("2025-04-06_23-27-41_806404" "2025-04-07_00-10-55_874793" "2025-04-07_00-53-29_237305" "2025-04-06_23-27-49_952763" "2025-04-07_00-06-24_195623" "2025-04-07_00-39-26_950682" "2025-04-06_23-27-18_083417" "2025-04-07_00-10-18_115604" "2025-04-07_00-52-53_764863" "2025-04-06_23-27-33_262686" "2025-04-07_00-05-59_335372" "2025-04-07_00-38-54_015738")
k_lst=("512,512,16" "512,512,32" "512,512,64" "512,512,128" "512,512,256" "512,512,512" "512,512,16" "512,512,32" "512,512,64" "512,512,128" "512,512,256" "512,512,512")
placement_lst=("pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "pre-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax" "post-softmax")

# Get the length of the first list
length=${#timestamp_lst[@]}

# Iterate over both lists
for (( i=0; i<length; i++ )); do
    curr_threhsh_subdir_name=${MODEL}_${DATASET}_placement-${placement_lst[i]}_k${k_lst[i]}
    curr_output_path=$OUTPUT_PATH/$curr_threhsh_subdir_name
    mkdir -p $curr_output_path
    cp $INPUT_PATH/${timestamp_lst[i]}/th.txt $curr_output_path/
done