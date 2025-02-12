# First of all create a unique time stamp for the current run
from datetime import datetime
timestamp = str(datetime.now()).replace(' ','_').replace('.','_').replace(':','-')
import itertools
import os
import logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename=f"logs/{timestamp}-app.log", filemode='w', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_dataset
from tqdm import tqdm
from human_eval.data import write_jsonl
from topk_llama import update_model, set_params
from util import generate_one_completion, get_model_names, get_model_num_attn_layers, compress_files_parallel


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--llama',
    type=str,
    default='34',
    choices=['2-7','3-8','3-8i','34','2-70','3-70','3-70i'],
    help='llama model alias to be loaded. specifying "i" after the number implies "Instruct-finetuned"')
parser.add_argument(
    '--mode',
    type=str,
    default='3',
    choices=['0','1','3'],
    help='0 - TH, 1 - TOPK, other number - BASELINE')
parser.add_argument(
    '--k',
    metavar='K',
    type=int,
    default=256,
    help='k parameter of top-k/threshold, effective only for --mode 0 or --mode 1')
parser.add_argument(
    '--calib_load_path',
    type=str,
    default="",
    help='path to a directory containing th.txt or sdc.txt (applicable '
         'only for mode=0 or when --sdc=offline-calibrated is used) If '
         'this argument is specified, then calibration will not take '
         'place, but rather loaded from the existing files.')
parser.add_argument(
    "--calib_tac", 
    help="Top-K At Calibrating. This would incorporate the "
         "sparsification effect already during the calibration",
    action="store_true")
parser.add_argument(
    '--calib_add_sigma',
    type=float,
    default=0.0,
    help='this amount of standard deviations will be added to the '
         'average threshold when at the end of the calibration the '
         'aggregating of the threshold obained from the various '
         'calibraion samples takes place. Has effect only for mode=0.')
parser.add_argument(
    '--calib_sample_frac',
    type=float,
    default=0.1,
    help='percentage of queries (i.e. attention rows) to use for calibration')
parser.add_argument(
    '--layerk',
    metavar='DICT',
    type=str,
    default='',
    help='dictionary that overwrites the --k for chosen layers. E.g 3:128 '
         'forces the 3rd layer to use k=128')
parser.add_argument(
    '--dataset',
    type=str,
    default='openai_humaneval',
    choices=['openai_humaneval'],
    help='evaluation dataset for generative decoding')
parser.add_argument(
    '--num_tasks',
    type=int,
    default=None,
    help='number of tasks to use for the evaluation. Default is using all tasks '
         'in the dataset.')
parser.add_argument(
    '--num_samples_per_task',
    type=int,
    default=1,
    help='number of completion answer to sample per every task in the dataset.')
parser.add_argument(
    '--max_seq_len',
    type=int,
    default=4096,
    help='maximum number of tokens allowed (prompt + completion)')
parser.add_argument(
    "--timestamps", 
    help="append timestamps to the filename of the cache-database (deleted at "
         "the end of the run) and to the products directory name. Always use "
         "this option when running multiple evaluations in parallel, in order "
         "to avoid database collisions.",
    action="store_true")
parser.add_argument(
    '--placement',
    type=str,
    default='pre-softmax',
    choices=['pre-softmax', 'post-softmax', 'none'],
    help='where to apply the top-k/threshold. none - stands for "do not apply"')
parser.add_argument(
    '--sdc',
    type=str,
    default='none',
    choices=['none', 'exact', 'exp-threshold', 'offline-calibrated'],
    help='software denominator compensation: method of compensating the softmax '
         'denominator, which becomes incomplete as a result of top-k/thresholding '
         'applied before the softmax. none - do nothing, exact - compute precisely '
         'the missing denominator term, exp-threshold applies only for the '
         'thresholding method, the two calibrated methods apply both for topk '
         'and thresholding.')
parser.add_argument(
    '--sdc_scale',
    type=float,
    default=1.0,
    help='multiplicative factor applied to the missing term, which is added by '
         'the compensation set by --sdc option. Has no effect when --sdc is none')    
parser.add_argument(
    "--vmc", 
    help="v-mean compensation. Adds a mean-row of V matrix (average across V's columns) "
         "to every output vector of the self_attention. The addition is weighted by "
         "(1-softmax_row.sum()). This option only has effect when the attention rows do not "
         "sum up to 1. Therefore, it is worth applying vmc only with --mode 0 or 1. Note that "
         "when --placement is set to 'pre-softmax' then it is necessary to apply --sdc "
         "different than 'none' in order to have the softmax output summed up to less than 1.",
    action="store_true")
parser.add_argument(
    "--prompt_prefix",
    type=str,
    default='',
    help="Constant text to prepend before each prompt in the evaluation"
)
parser.add_argument(
    "--prompt_suffix",
    type=str,
    default='',
    help="Constant text to append after each prompt in the evaluation"
)
parser.add_argument(
    "--capk", 
    help="cap the number of elements that pass the trhesholding to at most k in "
         " every attention row, prioritizing the last (most recent) tokens. "
         "Applicable only in mode=0.",
    action="store_true")
parser.add_argument(
    "--do_sample", 
    help="perform sampling of tokens rather than greedy",
    action="store_true")

# Check validity of cmd-line arguments
args = parser.parse_args()
assert(args.mode not in [0,1] or args.k >= 1)
assert(args.num_tasks is None or args.num_tasks >= 1)
assert(args.num_samples_per_task >= 1)
assert(args.max_seq_len >= 128)
assert(args.mode not in ['0','1'] or args.placement == 'pre-softmax' or args.sdc == 'none') # softmax compensation should be only when top-k/th is applied and only if it's applied pre-softmax
assert(args.mode == '0' or args.sdc != 'exp-threshold') # softmax compensation of exponentiated threshold can be only applied jointly with top-threshold pre-softmax
assert(not args.vmc or args.mode in ['0','1']) # v-mean compesation is applicatble only when sparsification is applied (mode 0 or 1)
assert(not (args.vmc and args.mode in ['0','1'] and args.placement == 'pre-softmax') or args.sdc != 'none') # v-mean compesation in a pre-softmax sparsifiation placement requires having a software denominator compensation to remain meaningful (without sdc: 1-softmax_row.sum() == 0)
assert(not(args.calib_tac) or args.mode == '0')  # topk-at-calibration should be applied only on mode=0 (thresholding)
assert(not(args.calib_load_path!="") or args.mode == '0' or args.sdc == 'offline-calibrated')
assert(not(args.capk) or args.mode == '0')  # --capk can be used only with top-threshld (mode=0)
assert(0.0 <= args.calib_sample_frac <= 1.0)
assert(args.num_samples_per_task == 1 or args.do_sample) # number of samples > 1 makes sense only when do_sample is true

print(f"RUN TIMESTAMP:{timestamp}")

# Pre-process arguments, create directories
if not args.timestamps:
    timestamp = ""
num_attn_layers = get_model_num_attn_layers(args.llama)
layer_k_dict = dict(eval('{'+args.layerk+'}'))
placement = args.placement if args.mode in {'0','1'} else 'none'
mode = int(args.mode)
model_name, model_shortname = get_model_names(args.llama)
cache_prefix_path = f"lm_cache/{model_name}{timestamp}"
products_dir_path = f"products/{timestamp}"
os.makedirs(products_dir_path, exist_ok=True)
os.makedirs("lm_cache", exist_ok=True)
os.makedirs("results-Llama", exist_ok=True)
device="cuda"
device_map_option="auto"
torch_dtype=torch.float16

# Configure Backend 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
set_seed(42) # for reproducibility

# Load the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map_option)
# print("Torch-compiling model...") # may speedup the model, however we didn't see a significant improvement. TODO: postpone this compilation to after the update_mode()
# model = torch.compile(model)
# print("Torch-compiling model - done.")

# Load the dataset
if args.dataset == "openai_humaneval":
    from human_eval.data import HUMAN_EVAL, read_problems
    problems = { int(task_id_str.split('/')[-1]): task_ for task_id_str, task_ in read_problems(HUMAN_EVAL).items() }
    num_tasks = min(len(problems), args.num_tasks) if args.num_tasks is not None else len(problems)
    calibration_requests = min(num_tasks*args.num_samples_per_task, 20)
else:
    dataset = load_dataset(args.dataset)
    problems = dataset["test"]
    num_tasks = min(len(problems.num_rows), args.num_tasks) if args.num_tasks is not None else len(problems.num_rows)
    calibration_requests = min(num_tasks*args.num_samples_per_task, 16)

# Configure the model for generation
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.use_cache = True
model.generation_config.repetition_penalty = 1.0
model.generation_config.max_length=args.max_seq_len
model.generation_config.do_sample = args.do_sample 
if (model.generation_config.do_sample):
    model.generation_config.temperature = 0.5
    model.generation_config.top_p = 0.95
    model.generation_config.typical_p = 0.95

# Configure the model for top-k/threshold
K_list=list([args.k]*num_attn_layers) if mode in [0,1] else [999,]
for key, value in layer_k_dict.items():
    K_list[key]=value
print(K_list)
test_layer = None

# Update the vanilla model's LlamaDecoderLayer layers with top-k parameters
update_model(model.model, False, products_dir_path) # this one changes dataformat to float32 again

# Update the model with specific top-k/threshold parameters (K etc) or calibrate if needed
if args.calib_load_path != "":
    # load the parameters and the pre-reviously calibrated thresholds from a specified directory
    set_params(model.model, K=K_list, mode=mode, placement=placement,
                sdc=args.sdc, sdc_scale=args.sdc_scale, vmc=args.vmc,
                calib_load_path=args.calib_load_path,
                calibrate=False,
                capk=args.capk,
                test_layer=None)
elif mode==0 or args.sdc == "offline-calibrated":
    # calibration is needed prior to an evaluation
    set_params(model.model, K=K_list, mode=mode, placement=placement,
                sdc=args.sdc,
                sdc_scale=args.sdc_scale,
                vmc=args.vmc,
                calib_load_path="",
                calibrate=True, 
                calib_tac=args.calib_tac,
                calib_add_sigma=args.calib_add_sigma,     
                calib_sample_frac=args.calib_sample_frac,                  
                calibration_requests=calibration_requests, 
                capk=args.capk,
                test_layer=None)

    # Calibration loop: run the language models on <calibration_requests>
    calib_req_lst = list(itertools.product(range(num_tasks), range(args.num_samples_per_task)))[:calibration_requests]
    for task_id_num, sample_id in tqdm(calib_req_lst, desc="Calibrating"):
        # trigger a round calibration (model's attention layers are now configured with calibrate=True)
        prompt = problems[task_id_num]["prompt"]
        generate_one_completion(tokenizer, model, prompt, args.prompt_prefix, args.prompt_suffix, products_dir_path=None)

# Set final configuration of the model for the inference
set_params(model.model, K=K_list, mode=mode, placement=placement,
            sdc=args.sdc,
            sdc_scale=args.sdc_scale,
            vmc=args.vmc,
            calib_load_path="",
            calibrate=False, 
            calib_tac=args.calib_tac,
            calib_add_sigma=args.calib_add_sigma,                       
            calib_sample_frac=args.calib_sample_frac,
            calibration_requests=calibration_requests, 
            capk=args.capk,
            test_layer=None)


# Run the language models on each of the dataset's test prompts
samples = []
for task_id_num, sample_id in tqdm(iterable=list(itertools.product(range(num_tasks), 
                                                          range(args.num_samples_per_task))),
                                   desc="Generating"):
    prompt = problems[task_id_num]["prompt"]
    prompt_and_completion = generate_one_completion(tokenizer, model, prompt, args.prompt_prefix, args.prompt_suffix, products_dir_path)
    samples.append(dict(task_id=problems[task_id_num]["task_id"], completion=prompt_and_completion))

# Write the generated samples to a file
samples_filename = f"{products_dir_path}/samples.jsonl"
print(f"Saving generated text samples to {samples_filename}")
write_jsonl(samples_filename, samples)


# TODO: integrate human_eval code (with our patches) or apply a patch to a pulled repo
print(f"Evaluating generated text samples")
if args.dataset == "openai_humaneval":
    from human_eval.evaluation import evaluate_functional_correctness
    pass_at_k = evaluate_functional_correctness(sample_file=samples_filename, 
                                                k=list(range(1, args.num_samples_per_task + 1)), 
                                                n_workers=32, 
                                                timeout=3, 
                                                problem_file=HUMAN_EVAL)
else:
    print("Warning: evaluation for datasets other than 'openai_humaneval' is not implemented yet")

# Print the aggregated evaluation scores to results-*/*.txt file
with open(f'results-Llama/{model_shortname}_{args.dataset}_mode{mode}_placement{placement}.txt', 'a') as f:
    f.write(timestamp + '\n')
    f.write(f'K:{K_list} mode:{mode} layer:{test_layer} placement:{placement} sdc:{args.sdc} sdc_scale:{args.sdc_scale} vmc:{args.capk} capk:{args.capk} calib_tac:{args.calib_tac} calib_add_sigma:{args.calib_add_sigma} calib_sample_frac:{args.calib_sample_frac} calib_load_path:{args.calib_load_path}\n')
    f.write(f'dataset:{args.dataset} num_tasks:{num_tasks} num_samples_per_task:{args.num_samples_per_task} max_seq_len:{args.max_seq_len} prompt_prefix:' + repr(args.prompt_prefix) + ' prompt_suffix:' + repr(args.prompt_suffix) + '\n')
    f.write('| Metric | Score |\n')
    f.write('|--------|-------|\n')
    for metric, score in pass_at_k.items():
        f.write(f'|{metric:8}|{score:1.4f} |\n')
    f.write('\n')


# Compress the products if any were produced (starting with "layer" and ending with "txt" or "csv")
layer_csv_txt_files_paths = glob.glob(os.path.join(products_dir_path, 'layer*.[ct][sx][vt]'))
compress_files_parallel(layer_csv_txt_files_paths, remove_uncompressed=True, num_cores=164)
