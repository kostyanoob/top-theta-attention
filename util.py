import lm_eval
import re
import string
import random
import gzip
import os
import multiprocessing

# ------- Functions for LM Eval Harness Framework--------

def get_model(type, model_name, device, dtype, batch_size, device_map_option):
    if type in ['causal', 'seq2seq']:
        return lm_eval.models.huggingface.HFLM(model_name, 
                                               device=device, 
                                               parallelize=True,
                                               dtype=dtype, 
                                               batch_size=batch_size, 
                                               trust_remote_code=bool(type=='causal'))
    else:
        raise ValueError("Should be causal or seq2seq")
    

# ------- Functions for non-LM Eval Harness Framework, for generative decoding -------------- 
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Union

# Define a function to generate text for a single prompt
def generate_one_completion(tokenizer: AutoTokenizer, 
                            model: AutoModelForCausalLM, 
                            prompt_text: str,  
                            prompt_prefix:str="", 
                            prompt_suffix:str="",
                            products_dir_path: Union[str,None]=None,
                            return_completion_tokens_only:bool=False) -> str:
    """
    generates text based on the given prompt_text

    Args:
        tokenizer - tokenizer model to break the prompt string into tokens, 
                    embed them and decode from embeddings back to human 
                    readable string.
        model - hugging face model, that supports.generate( )method
        prompt_text - str - input text to produce the completion for
        prompt_prefix: str - optional text to prepend before the prompt_text
        prompt_suffix: str - optional text to append after the prompt_text
        prompt_prefix: str - optional text to prepend before the prompt_text
        prompt_suffix: str - optional text to append after the prompt_text
        products_dir_path: str - path to a directory where the csv file with 
                           prompt lengths should be created. if None - don't
                           create such a file.
        return_completion_tokens_only: False - return both prompt and completion text
                                       Tru e- return text that originated from the completion tokens only                           

    Returns: a concatenation of {prompt, generated} strings. Without the 
             prefix/suffix if provided
    Returns: a concatenation of {prompt, generated} strings. Without the 
             prefix/suffix if provided
    """
    # modified_prompt = prefix + prompt + suffix
    modified_prompt_text = prompt_prefix + prompt_text + prompt_suffix

    # tokenize the modified prompt
    modified_prompt_ids = tokenizer.encode(modified_prompt_text, padding=True, return_tensors="pt").to(model.device)

    # Run the model to get the generated tokens (use the special terminator tokens - important to stop repetitive garbage generation)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    modified_prompt_and_completion_ids = model.generate(modified_prompt_ids, eos_token_id=terminators)
    
    # Get rid of the prefix and suffix that were used engineering the prompt in this function
    completion_text = tokenizer.decode(modified_prompt_and_completion_ids[0][len(modified_prompt_ids[0]):], skip_special_tokens=True)
    prompt_and_completion_text = prompt_text + completion_text # excludes prefix and suffix
    # modified_prompt = prefix + prompt + suffix
    modified_prompt_text = prompt_prefix + prompt_text + prompt_suffix

    # tokenize the modified prompt
    modified_prompt_ids = tokenizer.encode(modified_prompt_text, padding=True, return_tensors="pt").to(model.device)

    # Run the model to get the generated tokens (use the special terminator tokens - important to stop repetitive garbage generation)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    modified_prompt_and_completion_ids = model.generate(modified_prompt_ids, eos_token_id=terminators)
    
    # Get rid of the prefix and suffix that were used engineering the prompt in this function
    completion_text = tokenizer.decode(modified_prompt_and_completion_ids[0][len(modified_prompt_ids[0]):], skip_special_tokens=True)
    prompt_and_completion_text = prompt_text + completion_text # excludes prefix and suffix

    # write (prompt-length, generated-length) to a file
    if products_dir_path is not None:
        with open(f"{products_dir_path}/prompt_completion_lengths_per_sample.csv",'a') as f:
            for b_ in range(modified_prompt_ids.shape[0]):
                modified_prompt_len = len(modified_prompt_ids[b_])
                competion_len = len(modified_prompt_and_completion_ids[b_]) - len(modified_prompt_ids[b_])
                f.write(f'{modified_prompt_len} {competion_len}\n')
            for b_ in range(modified_prompt_ids.shape[0]):
                modified_prompt_len = len(modified_prompt_ids[b_])
                competion_len = len(modified_prompt_and_completion_ids[b_]) - len(modified_prompt_ids[b_])
                f.write(f'{modified_prompt_len} {competion_len}\n')

    return completion_text if return_completion_tokens_only else prompt_and_completion_text 


def  get_model_names(llama_alias: str) -> Tuple[str,str]:
    if llama_alias in ['2-7','2-70']:
        model_name = f"meta-llama/Llama-{llama_alias}b-hf" 
        model_shortname = f"Llama-{llama_alias}b-hf" 
    elif llama_alias=='34':
        model_name = f"codellama/CodeLlama-{llama_alias}b-hf"
        model_shortname = f"CodeLlama-{llama_alias}b-hf"
    elif llama_alias == '3-8':
        model_name = f"meta-llama/Meta-Llama-{llama_alias}B"
        model_shortname = f"Llama-{llama_alias}B"
    elif llama_alias == '3-8i':
        model_name = f"meta-llama/Meta-Llama-3-8B-Instruct"
        model_shortname = f"Llama-3-8B-Instruct"    
    elif llama_alias == '3.1-8i':
        model_name = f"meta-llama/Meta-Llama-3.1-8B-Instruct"
        model_shortname = f"Llama-3.1-8B-Instruct"            
    return model_name, model_shortname

def get_model_num_attn_layers(llama_alias: str) -> int:
    """
    return the number of attention layers in a model
    """
    return  80 if llama_alias in ['2-70', '3-70', '3-70i'] else \
            48 if llama_alias=='34' else \
            32 if llama_alias in ['3-8', '3-8i','3.1-8i','2-7'] else \
            0 #32 # Depends on the model 7B -> 32 layers


def gzip_one_file(in_file: str, remove_uncompressed:bool = False) -> bool:
    """compresses the in_file at and saves the compressed file under the same 
       name + .gz extension

    Parameters
    ----------
    in_file : str
        path to the input file to be compressed
    remove_uncompressed : bool, optional
        Trues <--> delete the original (uncompressed) file after the compressed 
        version has been saved, by default False

    Returns:
        True if and only if there was no problem compressing the file and no 
        problems removing it (when requested)
    """
    try:
        with open(in_file, 'rb') as f_in, gzip.open(in_file+'.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    except:
        print(f"Error: couldn't compress. Either {in_file} or {in_file}.gz are inaccessible.")
        return False

    if remove_uncompressed:
        try:
            os.remove(in_file)
        except:
            print(f"Info: couldn't remove the uncompressed {in_file} (file is inaccessible)")
            return False
    return True


def gzip_one_file_remove(in_file:str) -> bool:
    """
    compresses the file using gzip and removes the original uncompressed file

    Returns:
        True if and only if there was no problem compressing the file and no 
        problems removing it
    """
    return gzip_one_file(in_file, remove_uncompressed = True)


def gzip_one_file_keep(in_file:str) -> bool:
    """
    compresses the file using gzip and removes the original uncompressed file

    Returns:
        True if and only if there was no problem compressing the file
    """    
    return gzip_one_file(in_file, remove_uncompressed = False)


def compress_files_parallel(file_list:List[str], remove_uncompressed:bool = False, num_cores:int=16) -> None:
    """compresses all the files in the <file_list> at and saves each compressed file under the same 
       name + .gz extension. Each file is compressed in parallel to others, making use of <num_cores> cores.

    Parameters
    ----------
    file_list : List[str]
        list of paths path to the input files to be compressed
    remove_uncompressed : bool, optional
        Trues <--> delete the original (uncompressed) files after the compressed 
        version has been saved, by default False
    num_cores: int
        number of cores to parallelize the processing of the file lists.
    """
    print(f"Compressing {len(file_list)} products files using num_cores={num_cores}")

    gzipper_func = gzip_one_file_remove if remove_uncompressed else gzip_one_file_keep
    with multiprocessing.Pool(num_cores) as pool:
        return_statuses = pool.map(gzipper_func, file_list)
    
    if all(return_statuses):
        print("Done")
    else:
        print(f"Done with {len(return_statuses)-sum(return_statuses)} failures")

## --- From LongBench - scoring functions for result evaluations
from rouge import Rouge
import json
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def longbench_rouge_scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, rouge_score(prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def evaluate_longbench(samples_filename:str) -> Dict[str, float]:
    """
    evaluates the scores of a previously produced jsonl file
    returns a dictionary with the average rouge score of the
    generated responses
    
    """
    predictions, answers, lengths = [], [], []
    dataset = samples_filename.split('.')[0]
    with open(samples_filename, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            all_classes = data["all_classes"]
            if "length" in data:
                lengths.append(data["length"])
    score = longbench_rouge_scorer(dataset, predictions, answers, all_classes)
    return {'rouge': score}