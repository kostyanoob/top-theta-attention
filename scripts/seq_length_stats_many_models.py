import os
import csv
"""
This script prints a list of sequence length for a list of per-run product directories
"""


# List your directories here
PARENT_DIR_PATH="../products"
DIRS_LST = [
'2025-04-07_17-20-20_945838',  # qmsum baseline
'2025-04-23_11-57-33_398605',
'2025-04-25_22-18-45_839441',
'2025-04-27_19-48-14_444862',
'2025-05-08_11-03-24_380731',
'2025-04-17_10-33-00_960122',
'2025-04-17_03-28-42_874509',
'2025-04-19_13-21-14_271615',
'2025-05-08_11-03-31_414577',
'2025-04-23_11-57-25_414582',
'2025-04-23_20-30-24_263274',
'2025-04-24_05-01-46_863303',
'2025-04-24_13-31-42_308075',
'2025-04-14_16-40-59_439496',
'2025-04-15_05-08-13_701010',
'2025-04-16_08-43-59_678169',
'2025-04-16_15-55-41_567261',
'2025-04-09_17-34-28_654317',  # gov_report baseline
'2025-04-26_10-02-41_711830',
'2025-04-27_23-39-45_403838',
'2025-04-29_09-44-46_499998',
'2025-05-08_11-02-58_682523',
'2025-04-23_11-57-49_835361',
'2025-04-25_01-15-41_415189',
'2025-05-05_16-48-36_495207',
'2025-05-08_11-03-07_591731',
'2025-04-25_23-37-27_424025',
'2025-04-26_05-56-36_651144',
'2025-04-26_12-16-18_462819',
'2025-04-26_18-36-50_566293',
'2025-04-14_11-32-41_162279',
'2025-04-10_10-00-05_802004',
'2025-04-10_19-40-45_884053',
'2025-04-11_05-18-48_838271',
]

LABEL_LST = [
'Baseline qmsum     ',
'Top-θ pre- k=128   ',
'Top-θ pre- k=256   ',
'Top-θ pre- k=512   ',
'Top-θ pre- k=756   ',
'Top-θ post-k=128   ',
'Top-θ post-k=256   ',
'Top-θ post-k=512   ',
'Top-θ post-k=756   ',
'Top-k pre- k=128   ',
'Top-k pre- k=256   ',
'Top-k pre- k=512   ',
'Top-k pre- k=756   ',
'Top-k post-k=128   ',
'Top-k post-k=256   ',
'Top-k post-k=512   ',
'Top-k post-k=756   ',
'Baseline gov_report',
'Top-θ pre- k=128   ',
'Top-θ pre- k=256   ',
'Top-θ pre- k=512   ',
'Top-θ pre- k=756   ',
'Top-θ post-k=128   ',
'Top-θ post-k=256   ',
'Top-θ post-k=512   ',
'Top-θ post-k=756   ',
'Top-k pre- k=128   ',
'Top-k pre- k=256   ',
'Top-k pre- k=512   ',
'Top-k pre- k=756   ',
'Top-k post-k=128   ',
'Top-k post-k=256   ',
'Top-k post-k=512   ',
'Top-k post-k=756   ',
]

for directory, desc in zip(DIRS_LST, LABEL_LST):
    csv_path = os.path.join(PARENT_DIR_PATH, directory, 'prompt_completion_lengths_per_sample.csv')
    col1 = []
    col2 = []
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
            next(reader)  # Skip header
            for row in reader:
                # Remove empty strings that may result from multiple spaces
                row = [item for item in row if item]
                if len(row) >= 2:
                    try:
                        col1.append(float(row[0]))
                        col2.append(float(row[1]))
                    except ValueError:
                        continue  # Skip rows where conversion fails
        if col1 and col2:
            avg1 = sum(col1) / len(col1)
            avg2 = sum(col2) / len(col2)
            min2 = min(col2)
            max2 = max(col2)
            numbelow512_2 = len(list(filter(lambda x: x<512, col2)))
            print(f"{desc}: prompt = {avg1:.1f}, completion = avg:{avg2:.1f} min:{min2:.1f} max:{max2:.1f} num_below_512={numbelow512_2}")
        else:
            print(f"{csv_path}: no valid data in columns")
    except FileNotFoundError:
        print(f"{csv_path}: file not found")
