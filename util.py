#%%
import lm_eval
import re
import string
import random
import gzip
import os
import multiprocessing
#%%
# 
# 
# Following functions used only for initial testing of the T5 Enc-Dec model

def get_input(process_doc, task_name, task_object):
    input = None
    if task_name == 'openbookqa':
        input = process_doc["query"] + "\n" 
        for i, id  in enumerate(["(A)","(B)","(C)","(D)"]):
            input += id + " "
            input += process_doc["choices"][i]  + " "
        input+="\nAnswer:"
    elif task_name == 'squad2':
        input = process_doc['context'] + " \\n (" + process_doc['title'] + ") "
        input += process_doc['question'] + " Answer:"
    # elif task == 'hellaswag':
    #     input = process_doc["query"] + "\n\n" 
    #     for i, id  in enumerate(["(A)","(B)","(C)","(D)"]):
    #         if i>0:
    #             input += "\n(or)\n"
    #         input += process_doc["choices"][i]  + " "
    #     # input += "\n\nWhat is the right answer?\n\n"
    elif task_name == 'arc_easy':
        input = process_doc["query"]
        input = input[:-8] + "\n"
        num_choices = len(process_doc["choices"])
        for i, id  in enumerate(["(A)", "(B)", "(C)", "(D)", "(E)"]):
            if i < num_choices:
                input += id + " "
                input += process_doc["choices"][i]  + " "
        input += "Answer:"
    elif task_name=='race':
        # input = task_object.doc_to_text(process_doc)
        text = "Article: " + process_doc["article"] + "\n\n"
        for problem in process_doc["problems"][:-1]:
            if problem["question"][-6:] == "  _  .":
                text += (
                    problem["question"][-5:] + task_object.get_answer_option(problem) + "\n"
                )
            else:
                question = "Question: " + problem["question"] + "\n"
                answer = "Answer: " + task_object.get_answer_option(problem) + "\n"
                text += question + answer
        text += "Question: " + task_object.last_problem(process_doc)["question"] + "\n" # Added "Question:" prompt 
        input = text

        option_names = ['(A)','(B)','(C)','(D)']
        for i, option in enumerate(task_object.last_problem(process_doc)['options']):
            input += f" {option_names[i]} {option}"
        input += "\nAnswer: "
    else:
        raise ValueError(f"{task_name} not implemented")
    return ((input,{'until':[".","\n"]}))

#%% Functions to find the exact match scores
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)
    return ( pred == gold )

def find_exact_match_score(responses, task_docs, task, filename):
    exact_match = 0
    for i, prediction in enumerate(responses):
            # Openbookqa
            if task=='openbookqa' or task=='arc_easy':
                cmp = lm_eval.metrics.metric_max_over_ground_truths(
                    exact_match_score, prediction.replace("\n",""), [task_docs[i]['choices'][task_docs[i]['gold']]])

            # Squad
            # print(i, prediction, task_docs[i]['answers']['text'])
            elif task=='squad2':
                cmp=False
                if task_docs[i]['answers']['text']==[]:
                    if prediction=="no answer>":
                        cmp = True
                else:
                    cmp = lm_eval.metrics.metric_max_over_ground_truths(
                    exact_match_score, prediction.replace("\n",""), task_docs[i]['answers']['text'])
            elif task=='race':
                letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
                prediction_text = prediction
                for s in ['(A)','(B)','(C)','(D)']:
                    prediction_text=prediction_text.replace(s,"")
                options = task_docs[i]['problems'][-1]['options']
                ans = task_docs[i]['problems'][-1]['answer']
                num = letter_to_num[ans]
                cmp = lm_eval.metrics.metric_max_over_ground_truths(
                    exact_match_score, prediction_text.replace("\n",""), [options[num]])
                cmp = cmp or lm_eval.metrics.metric_max_over_ground_truths(
                    exact_match_score, prediction.replace("\n",""), [ans])
            else:
                raise ValueError(f'{task} not Implemented')

            exact_match += cmp
            # with open(filename,'a') as f:
            #     # f.write(str(cmp) + '\n')
            #     f.write(f'{str(cmp)} pred:{prediction} gold:{options[num]}\n')

    return exact_match * 100.0 / len(responses)

# %%
def get_requests(seed, num_samples, task_name):
    # tasks_list = ['ai2_arc', 'openbookqa', 'race', 'squad2']
    tasks_list =  [task_name]

    task_dict = lm_eval.tasks.get_task_dict(tasks_list)
    task_object = task_dict[task_name]
    
    print("Has Test Set ? -",task_object.has_test_docs())
    task_doc_func = task_object.validation_docs
    task_docs = list(task_doc_func())
    rnd = random.Random()
    rnd.seed(seed)
    rnd.shuffle(task_docs)

    num = num_samples if num_samples != -1 else len(task_docs)
    
    requests=[]
    for i in range(num):
        process_doc = task_docs[i]
        requests.append( get_input(process_doc=process_doc, task_name=task_name, task_object=task_object) )
    return (num, requests, task_docs[:num])

#%%
def generate(model, request):
    responses = model.greedy_until(request)
    return responses

# Above functions used only only for initial testing for the Enc-Dec T5 model





# -------Functions for LM Eval Harness Framework--------

def get_model(type, model_name, device, dtype, batch_size, device_map_option):
    if type in ['causal', 'seq2seq']:
        return lm_eval.models.huggingface.HFLM(model_name, 
                                               device=device, 
                                               parallelize=True,
                                               dtype=dtype, 
                                               batch_size=batch_size, 
                                               device_map_option=device_map_option,
                                               trust_remote_code=bool(type=='causal'))
    else:
        raise ValueError("Should be causal or seq2seq")
    

# -------Functions for non-LM Eval Harness Framework, for generative decoding -------------- 
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Union

# Define a function to generate text for a single prompt
def generate_one_completion(tokenizer: AutoTokenizer, 
                            model: AutoModelForCausalLM, 
                            prompt_text: str,  
                            prompt_prefix:str="", 
                            prompt_suffix:str="",
                            products_dir_path: Union[str,None]=None) -> str:
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
        products_dir_path: str - path to a directory where the csv file with 
                           prompt lengths should be created. if None - don't
                           create such a file.

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

    # write (prompt-length, generated-length) to a file
    if products_dir_path is not None:
        with open(f"{products_dir_path}/prompt_completion_lengths_per_sample.csv",'a') as f:
            for b_ in range(modified_prompt_ids.shape[0]):
                modified_prompt_len = len(modified_prompt_ids[b_])
                competion_len = len(modified_prompt_and_completion_ids[b_]) - len(modified_prompt_ids[b_])
                f.write(f'{modified_prompt_len} {competion_len}\n')

    return prompt_and_completion_text

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
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        model_shortname = "Llama-3-8B-Instruct"        
    elif llama_alias == '3-70':
        model_name = f"meta-llama/Meta-Llama-{llama_alias}B"
        model_shortname = f"Llama-{llama_alias}B"
    elif llama_alias == '3-70i':
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        model_shortname = "Llama-3-70B-Instruct"                
    return model_name, model_shortname

def get_model_num_attn_layers(llama_alias: str) -> int:
    """
    return the number of attention layers in a model
    """
    return  80 if llama_alias in ['2-70', '3-70', '3-70i'] else \
            48 if llama_alias=='34' else \
            32 if llama_alias in ['3-8', '3-8i','2-7'] else \
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
