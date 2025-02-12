"""

Converts the original thresholds file (th.txt) into a clean compressed csv.

Input:  txt file contiaing the calibrated thresholds for all layers, heads, seqlens.
        The file is space-delimited, without headers. Each line has the pattern:
        L<layer>_H<head>:<seq_len> <threshold> <num_samples> <k>

Output: a new, compressed csv file, with 4 columns: 'layer', 'head', 'seq_len', 'threshold'
        (header is imprinted)

Usage:
python th_txt_to_csv_gz.py <PATH_TO_INPUT_th.txt> <PATH_TO_OUTPUT_csv.gz>

"""
import sys
import csv
import gzip

# Read the input and output file paths from command line arguments
input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Open the input file for reading
with open(input_file_path, 'r') as input_file:
    # Read all the lines from the input file
    lines = input_file.readlines()

# Initialize an empty list to store the output data
output_data = []

# Iterate through each line in the input file
for line in lines:
    # Use regular expression to extract the numbers from the line
    line_lst = line[1:].split(" ")
    layer, headseqlen = line_lst[0].split("_H")
    head, seqlen = headseqlen.split(":")
    threshold = line_lst[1]

    # Create a list of values for the current row
    row = [int(layer), int(head), int(seqlen), float(threshold)]

    # Append the row to the output data list
    output_data.append(row)

# Open the output file for writing
with gzip.open(output_file_path, 'wt', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write the header row
    csv_writer.writerow(['layer', 'head', 'seq_len', 'threshold'])

    # Write the output data rows
    csv_writer.writerows(output_data)

print('Output file written successfully.')