import glob
import gzip
import os
import csv
import pandas as pd
import numpy as np
import multiprocessing
from typing import Tuple, Union, List
from collections import namedtuple
from tqdm import tqdm


ResultMeta = namedtuple('ResultEntry', ['timestamp', 'label', 'k'])
ResultMetrics = namedtuple('ResultMetrics', ['timestamp', 'model_name','accuracy_mean', 'accuracy_std', 'kv_seq_len_mean', 'kv_seq_len_std', 'size_mean', 'size_std', 'fullsize_mean', 'fullsize_std'])


def aggregate_accuracy_and_size_of_many_runs(results_meta_list:List[ResultMeta], 
                                         accuracy_metric:str,
                                         size_metric:str,
                                         inference_phase:str, 
                                         results_dir_path:str, 
                                         products_dir_path:str,
                                         num_cores:int=1) -> pd.DataFrame:
    """
    Returns a pandas dataframe with metrics. Each row in the dataframe will correspond to an entry in results_meta_list.
    """
    if num_cores > 1:
        # Read the results in parallel
        print(f"Reading results in parallel using {num_cores} parallel processes")
        with multiprocessing.Pool(num_cores) as pool:
            accuracy_arg_list = [(accuracy_metric, r.timestamp, results_dir_path) for r in results_meta_list]
            accuracy_metric_tuples = pool.map(find_accuracy_mapper, accuracy_arg_list)

            size_arg_list = [(f"{products_dir_path}/{r.timestamp}", size_metric, inference_phase, False) for r in results_meta_list]
            size_metric_tuples = pool.map(summarize_products_metric_mapper, size_arg_list)

        results_metrics_list = []
        # join the accuracy and the size metrcs
        for r, accuracy_metric_tuple, size_metric_tuple in zip(results_meta_list, accuracy_metric_tuples, size_metric_tuples):
            model_name, accuracy_mean, accuracy_std = accuracy_metric_tuple
            kv_seq_len_mean, kv_seq_len_std, size_mean, size_std, fullsize_mean, fullsize_std = size_metric_tuple
            results_metrics_list.append(ResultMetrics(timestamp=r.timestamp, 
                                                    model_name=model_name,
                                                    accuracy_mean=accuracy_mean, 
                                                    accuracy_std=accuracy_std, 
                                                    kv_seq_len_mean=kv_seq_len_mean, 
                                                    kv_seq_len_std=kv_seq_len_std, 
                                                    size_mean=size_mean, 
                                                    size_std=size_std, 
                                                    fullsize_mean=fullsize_mean, 
                                                    fullsize_std=fullsize_std))            

    else:
        results_metrics_list = []
        for r in tqdm(results_meta_list, desc="Reading results"):
            
            # parse the ACCURACY METRIC (acc_norm/pass@1) metric of the given run (identified by the timestamp) from the entire directory
            model_name, accuracy_mean, accuracy_std = find_accuracy(accuracy_metric, r.timestamp, results_dir_path)    

            # parse the SIZE METRIC (kept_attn/kept_vrow)
            kv_seq_len_mean, kv_seq_len_std, size_mean, size_std, fullsize_mean, fullsize_std = summarize_products_metric(f"{products_dir_path}/{r.timestamp}", 
                                                                                                                        size_metric, 
                                                                                                                        inference_phase)
            # join the accuracy and the size metrcs
            results_metrics_list.append(ResultMetrics(timestamp=r.timestamp, 
                                                    model_name=model_name,
                                                    accuracy_mean=accuracy_mean, 
                                                    accuracy_std=accuracy_std, 
                                                    kv_seq_len_mean=kv_seq_len_mean, 
                                                    kv_seq_len_std=kv_seq_len_std, 
                                                    size_mean=size_mean, 
                                                    size_std=size_std, 
                                                    fullsize_mean=fullsize_mean, 
                                                    fullsize_std=fullsize_std))

    # Merge the accuracy-metric columns and the size-metrics columns
    meta_df=pd.DataFrame(results_meta_list)
    metrics_df=pd.DataFrame(results_metrics_list)
    assert(not metrics_df.isnull().values.any()),"Error: some metrics were not gathered. Check that for all the timestamps you've specified in RESULTS_META_LIST - results and products are accessible"
    df = meta_df.merge(metrics_df, on='timestamp', how='left')

    # Add metrics relaed information
    df['inference_phase'] = inference_phase
    df['accuracy_metric'] = accuracy_metric
    df['size_metric'] = size_metric

    assert(len(df) == len(meta_df)),"Error: Some information was lost in the merging between the meta information and the metrics"
    assert(not df.isnull().values.any()),"Error: NaN exists in the dataframe (meta_df,metrics_df)"

    return df

def find_accuracy_mapper(arg: Tuple[str,str,str]) -> Union[Tuple[str, float, float], Tuple[None,None,None]]:
    return find_accuracy(*arg)

def find_accuracy( accuracy_metric:str, timestamp:str, results_dir_path:str) -> Union[Tuple[str, float, float], Tuple[None,None,None]]:
    if accuracy_metric=='acc_norm':
        model_name, accuracy_mean, accuracy_std = find_acc_norm(timestamp, results_dir_path) 
    elif accuracy_metric=='pass@1':
        model_name, accuracy_mean, accuracy_std = find_passat1(timestamp, results_dir_path) 
    else:
        model_name, accuracy_mean, accuracy_std = None, None, None

    return model_name, accuracy_mean, accuracy_std

def find_passat1(timestamp:str, dir_path:str) -> Union[Tuple[str, float, float], Tuple[None, None,None]]:
    '''
    Searches in many results (txt) filed for a paragraph describing 
    the acc_nom of a given timestamp run.

    Return: 3-tuple: (model_name, acc_norm average, acc_norm std)
    '''
    
    matching_files = glob.glob(f"{dir_path}/*.txt")
    for file_path in matching_files:
        ret_tuple = find_passat1_in_file(timestamp, file_path)
        if ret_tuple != (None, None):
            model_name = os.path.basename(file_path).split("_")[0]
            return (model_name,) + ret_tuple

    print(f"Error: couldn't extract pass@1 for timestamp {timestamp} in any of the files within the directory {dir_path}")
    return None, None, None


def find_passat1_in_file(timestamp:str, file_path:str, verbose:bool=False) -> Union[Tuple[float, float], Tuple[None, None]]:
    """
    Read a text file at <file_path> and find a paraagraph that begins with the timestamp

    Example: when a file contains a paragraph like this:
    2024-12-12_14-52-55_962978
    K:[512, 512, 64, 64] mode:1 layer:None placement:pre-softmax
    dataset:openai_humaneval num_tasks:164 num_samples_per_task:1 max_seq_len:2048
    | Metric | Score |
    |--------|-------|
    |pass@1  |0.2439 |

    Expected output is (0.2439, 0)

    Return: 2-tuple: (pass@1 average, pass@1 std)

    """
    METRIC = 'pass@1'

    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into paragraphs
    paragraphs = content.split('\n\n')

    # Find the paragraph that starts with the given timestamp
    target_paragraph = None
    for paragraph in paragraphs:
        if paragraph.strip().startswith(timestamp):
            target_paragraph = paragraph
            break

    if not target_paragraph:
        if verbose:
            print("Error: file {file_path} doesn't contain the required timestamp {timestamp}")
        return None, None  # Timestamp not found

    # Find the acc_norm line in the paragraph
    target_paragraph_lines = list(map(str.strip,filter(len,target_paragraph.split('\n'))))
    columns = list(map(str.strip,target_paragraph_lines[-3].split('|')))
    target_paragraph_last_row_values = target_paragraph_lines[-1].split('|')

    if len(target_paragraph_last_row_values) != len(columns):
        if verbose:
            print("Error: the last row of the paragraph is not equal in length to the column-name line")
        return None, None
    elif METRIC not in map(str.strip,target_paragraph_last_row_values):
        if verbose:
            print(f"Error: the last row of the paragraph does not contain the {METRIC} metric")
        return None, None        
    else:
        acc_norm = float(target_paragraph_last_row_values[columns.index('Score')])
        acc_norm_stderr = 0
        return acc_norm, acc_norm_stderr 


def find_acc_norm(timestamp:str, dir_path:str) -> Union[Tuple[float, float], Tuple[None,None]]:
    '''
    Searches in many results (txt) filed for a paragraph describing 
    the acc_nom of a given timestamp run.

    Return: 3-tuple: (model_name, acc_norm average, acc_norm std)
    '''
    # 
    
    matching_files = glob.glob(f"{dir_path}/*.txt")
    for file_path in matching_files:
        ret_tuple = find_acc_norm_in_file(timestamp, file_path)
        if ret_tuple != (None, None):
            model_name = os.path.basename(file_path).split("_")[0]
            return (model_name,) + ret_tuple
    
    print(f"Error: couldn't extract acc_norm for timestamp {timestamp} in any of the files within the directory {dir_path}")
    return None, None, None


def find_acc_norm_in_file(timestamp:str, file_path:str, verbose:bool=False) -> Union[Tuple[float, float], Tuple[None, None]]:
    """
    Read a text file at <file_path> and find a paraagraph that begins with the timestamp

    Example: when a file contains a paragraph like this:
    2024-12-05_14-10-15_821283
    K:[512, 512, 64, 64, 64, 64] mode:1 layer:None placement:pre-softmax 
    |    Tasks    |Version|Filter|n-shot| Metric |Value|   |Stderr|
    |-------------|-------|------|-----:|--------|----:|---|-----:|
    |arc_challenge|Yaml   |none  |    25|acc     | 0.44|±  |0.1013|
    |             |       |none  |    25|acc_norm| 0.52|±  |0.1020|

    Expected output is (0.52, 0.1020)

    Return: 2-tuple: (acc_norm average, acc_norm std)

    """

    METRIC = 'acc_norm'

    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into paragraphs
    paragraphs = content.split('\n\n')

    # Find the paragraph that starts with the given timestamp
    target_paragraph = None
    for paragraph in paragraphs:
        if paragraph.strip().startswith(timestamp):
            target_paragraph = paragraph
            break

    if not target_paragraph:
        if verbose:
            print("Error: file {file_path} doesn't contain the required timestamp {timestamp}")
        return None, None  # Timestamp not found

    # Find the acc_norm line in the paragraph
    target_paragraph_lines = list(map(str.strip,filter(len,target_paragraph.split('\n'))))
    columns = list(map(str.strip,target_paragraph_lines[-4].split('|')))
    target_paragraph_last_row_values = target_paragraph_lines[-1].split('|')

    if len(target_paragraph_last_row_values) != len(columns):
        if verbose:
            print("Error: the last row of the paragraph is not equal in length to the column-name line")
        return None, None
    elif METRIC not in map(str.strip,target_paragraph_last_row_values):
        if verbose:
            print(f"Error: the last row of the paragraph does not contain the {METRIC} metric")
        return None, None        
    else:
        acc_norm = float(target_paragraph_last_row_values[columns.index('Value')])
        acc_norm_stderr = float(target_paragraph_last_row_values[columns.index('Stderr')])
        return acc_norm, acc_norm_stderr  

def summarize_products_metric_mapper(arg:Tuple[str,str,str,bool]) -> Tuple[float, float, float, float, float, float]:
    return summarize_products_metric(*arg)

def summarize_products_metric(directory_path:str, metric:str, inference_phase:str, verbose:bool=False) -> Tuple[float, float, float, float, float, float]:
    """
    Summarize all kept_attn or kept_vrow csv files in a given directory. Return the average 
    across-(layers,head,input) number of kept attention elements and total 
    attention elements.

    Can deal with 2 cases of metric:
    1) metric="kept_attn"
    Each csv file contains headerless columns that correspond to 
    [layer, head, kv-seq-len, kept_attn_numel_per_head, full_attn_numel_one_head]. 
    The summarization means taking all the files (printing their names as well), 
    reading all their lines and returning 4 values: (average,std) of 
    kept_attn_numel_per_head and (average,std) of full_attn_numel_one_head columns.

    1) metric="kept_vrow"
    Each csv file contains headerless columns that correspond to 
    [layer, head, kv-seq-len, kept_numvrows_numel_per_head, full_numvvrows_one_head]. 
    The summarization means taking all the files (printing their names as well), 
    reading all their lines and returning 4 values: (average,std) of 
    kept_numvrows_numel_per_head and (average,std) of full_numvvrows_one_head columns.

    Arguments:
     - path to a directory with all the layer<i>_kept_attn_<inference_phase>.csv files
     - metric: either "kept_attn" or "kept_vrow"
     - file suffix that indicates the inference suffix. inside a directory the function 
       
    Returns:
    (kv_seq_len_mean, kv_seq_len_std, size_mean, size_std, fullsize_mean, fullsize_std)
    """

    assert metric in {"kept_attn", "kept_vrow"}

    # Pattern to match the files
    pattern_csv = os.path.join(directory_path, f"layer*_{metric}_{inference_phase}.csv")  
    pattern_csv_gz = os.path.join(directory_path, f"layer*_{metric}_{inference_phase}.csv.gz")

    
    # Find all matching files
    matching_files = glob.glob(pattern_csv) + glob.glob(pattern_csv_gz)
    
    if not matching_files:
        print("No matching files found.")
        return (None, None, None, None, None, None)
    
    kv_seq_len_values = []
    kept_attn_values = []
    full_attn_values = []
    
    if verbose:
        print("Processing files:")
    for file_path in matching_files:
        if verbose:
            print(f"- {os.path.basename(file_path)}")
        if file_path.endswith(".csv"):
            csvfile = open(file_path, 'r')
        elif file_path.endswith("csv.gz"):
            csvfile = gzip.open(file_path, 'rt', newline='')
        else:
            if verbose:
                print("Error at file:" + file_path)
            return (None, None, None, None, None, None)
        
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            kv_seq_len_values.append(float(row[2]))  # kv_seq_len
            kept_attn_values.append(float(row[3]))  # kept_attn_numel_per_head / kept_vrow_per_head
            full_attn_values.append(float(row[4]))  # full_attn_numel_one_head / all_vrow_per_head

        csvfile.close()
    
    # Calculate statistics
    kv_seq_len_mean = np.mean(kv_seq_len_values) 
    kv_seq_len_std = np.std(kv_seq_len_values) 
    size_mean = np.mean(kept_attn_values)
    size_std = np.std(kept_attn_values)
    fullsize_mean = np.mean(full_attn_values)
    fullsize_std = np.std(full_attn_values)

    
    return (kv_seq_len_mean, kv_seq_len_std, size_mean, size_std, fullsize_mean, fullsize_std)